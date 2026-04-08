# -*- coding: utf-8 -*-
from concurrent.futures import Executor, Future
from threading import Lock, Thread
import queue
import atexit
from collections import namedtuple
import time
import os

import multiprocess as mp

# 用于存储任务结果的数据结构
_TaskResult = namedtuple('_TaskResult', ['task_id', 'result', 'exception'])

class _WorkItem:
    def __init__(self, future, fn, args, kwargs):
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

def _process_worker(task_queue, result_queue, initializer, initargs, max_tasks_per_child):
    """工作进程函数"""
    # 执行初始化函数
    if initializer is not None:
        try:
            initializer(*initargs)
        except Exception:
            # 初始化失败，发送错误信号
            result_queue.put(_TaskResult(-999, None, Exception("Initializer failed")))
            return
    
    tasks_completed = 0
    
    while True:
        try:
            work_item = task_queue.get(timeout=1)
            if work_item is None:  # 结束信号
                break
            
            try:
                result = work_item.fn(*work_item.args, **work_item.kwargs)
                exception = None
            except Exception as exc:
                result = None
                exception = exc
            
            result_queue.put(_TaskResult(work_item.future._task_id, result, exception))
            
            tasks_completed += 1
            
            # 检查是否达到最大任务数限制
            if max_tasks_per_child is not None and tasks_completed >= max_tasks_per_child:
                break
                
        except queue.Empty:
            continue
        except Exception:
            # 捕获所有异常，防止工作进程崩溃
            continue

class ProcessPoolExecutor(Executor):
    def __init__(self, max_workers=None, mp_context=None,
                 initializer=None, initargs=(), *, max_tasks_per_child=None):
        """
        初始化进程池执行器
        
        Args:
            max_workers: 最大工作进程数，默认为CPU核心数
            mp_context: multiprocessing上下文对象
            initializer: 工作进程初始化函数
            initargs: 传递给初始化函数的参数元组
            max_tasks_per_child: 每个工作进程执行的最大任务数，超过后重启进程
        """
        if max_workers is None:
            self._max_workers = mp.cpu_count()
        else:
            if max_workers <= 0:
                raise ValueError("max_workers must be greater than 0")
            self._max_workers = max_workers
        
        # 设置multiprocessing上下文
        if mp_context is None:
            self._mp_context = mp
        else:
            self._mp_context = mp_context
        
        # 存储初始化参数
        self._initializer = initializer
        self._initargs = initargs
        self._max_tasks_per_child = max_tasks_per_child
        
        # 创建进程间通信队列
        self._task_queue = self._mp_context.Queue()
        self._result_queue = self._mp_context.Queue()
        
        # 进程列表
        self._processes = []
        
        # 任务ID计数器
        self._task_counter = 0
        self._task_counter_lock = Lock()
        
        # 存储活跃的Future对象
        self._futures = {}
        self._futures_lock = Lock()
        
        # 控制池状态
        self._shutdown = False
        self._shutdown_lock = Lock()
        
        # 启动工作进程
        self._start_processes()
        
        # 注册退出清理函数
        atexit.register(self.shutdown, wait=False)
    
    def _start_processes(self):
        """启动工作进程"""
        for i in range(self._max_workers):
            p = self._mp_context.Process(
                target=_process_worker,
                args=(
                    self._task_queue,
                    self._result_queue,
                    self._initializer,
                    self._initargs,
                    self._max_tasks_per_child
                )
            )
            p.start()
            self._processes.append(p)
        
        # 启动结果收集线程
        self._result_collector_thread = Thread(
            target=self._result_collector,
            daemon=True
        )
        self._result_collector_thread.start()
    
    def _get_next_task_id(self):
        """获取下一个任务ID"""
        with self._task_counter_lock:
            task_id = self._task_counter
            self._task_counter += 1
            return task_id
    
    def _result_collector(self):
        """结果收集线程函数"""
        while not self._shutdown:
            try:
                task_result = self._result_queue.get(timeout=0.5)
                
                # 特殊处理初始化失败
                if task_result.task_id == -999:
                    print(f"Worker initialization failed: {task_result.exception}")
                    continue
                
                with self._futures_lock:
                    future = self._futures.get(task_result.task_id)
                    if future:
                        if task_result.exception:
                            future.set_exception(task_result.exception)
                        else:
                            future.set_result(task_result.result)
                        
                        # 从活跃列表中移除
                        del self._futures[task_result.task_id]
            except queue.Empty:
                continue
            except Exception:
                # 捕获异常，防止线程崩溃
                continue
    
    def submit(self, fn, *args, **kwargs):
        """
        提交一个可调用对象以异步执行
        
        Args:
            fn: 要执行的可调用对象
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Future: 表示可调用对象执行的Future对象
        """
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("Cannot schedule new futures after shutdown")
            
            future = Future()
            task_id = self._get_next_task_id()
            future._task_id = task_id
            
            work_item = _WorkItem(future, fn, args, kwargs)
            
            with self._futures_lock:
                self._futures[task_id] = future
            
            self._task_queue.put(work_item)
            
            return future
    
    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """
        并行映射函数到迭代项
        
        Args:
            fn: 映射函数
            iterables: 可迭代对象
            timeout: 超时时间
            chunksize: 块大小（在此实现中不使用）
            
        Returns:
            生成器: 返回结果的生成器
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be a non-negative number or None")
        
        # 将迭代项打包成元组
        fs = [self.submit(fn, *args) for args in zip(*iterables)]
        
        # 收集结果
        results = []
        for f in fs:
            if timeout is not None:
                result = f.result(timeout=timeout)
            else:
                result = f.result()
            results.append(result)
        
        return results
    
    def shutdown(self, wait=True):
        """
        关闭进程池
        
        Args:
            wait: 是否等待任务完成
        """
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True
        
        # 发送结束信号给所有工作进程
        for _ in range(self._max_workers):
            self._task_queue.put(None)
        
        if wait:
            # 等待所有进程结束
            for p in self._processes:
                p.join()
        
        # 清理资源
        if hasattr(self, '_result_collector_thread'):
            self._result_collector_thread.join(timeout=1)

# 测试代码
if __name__ == "__main__":
    import time
    import os
    
    def test_initializer():
        """测试初始化函数"""
        print(f"Worker initialized in process {os.getpid()}")
        # 设置一些全局变量
        global worker_init_flag
        worker_init_flag = True
    
    def test_function(x):
        """测试函数"""
        print(f"Processing {x} in process {os.getpid()}")
        time.sleep(0.5)  # 模拟耗时操作
        return x * x
    
    def test_with_error(x):
        """测试错误处理"""
        if x == 3:
            raise ValueError("Intentional error for testing")
        return x * 2
    
    print("=== 测试带有初始化函数的 ProcessPoolExecutor ===")
    
    # 测试带初始化器的基本功能
    with ProcessPoolExecutor(max_workers=2, initializer=test_initializer) as executor:
        # 提交任务
        futures = [executor.submit(test_function, i) for i in range(5)]
        
        # 获取结果
        results = []
        for future in futures:
            try:
                result = future.result(timeout=10)
                results.append(result)
                print(f"Result: {result}")
            except Exception as e:
                print(f"Error: {e}")
        
        print(f"All results: {results}")
    
    print("\n=== 测试 max_tasks_per_child 功能 ===")
    with ProcessPoolExecutor(max_workers=2, max_tasks_per_child=2) as executor:
        futures = [executor.submit(test_function, i) for i in range(6)]
        
        results = []
        for future in futures:
            try:
                result = future.result(timeout=10)
                results.append(result)
                print(f"Result: {result}")
            except Exception as e:
                print(f"Error: {e}")
        
        print(f"Results with max_tasks_per_child: {results}")
    
    print("\n=== 测试错误处理 ===")
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(test_with_error, i) for i in range(5)]
        
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=5)
                print(f"Task {i}: {result}")
            except Exception as e:
                print(f"Task {i}: Error - {e}")
    
    print("\n=== 测试 map 功能 ===")
    with ProcessPoolExecutor(max_workers=3) as executor:
        numbers = list(range(1, 6))
        results = executor.map(lambda x: x ** 3, numbers)
        print(f"Map results: {list(results)}")
    
    print("\nAll tests completed!")
