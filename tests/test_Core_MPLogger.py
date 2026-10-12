# -*- coding: utf-8 -*-
"""QuantStudio MPLogger 多进程日志模块测试

使用方法:
    * 运行全部测试: python tests/test_Core_MPLogger.py
    * 运行指定测试: python -m unittest tests.test_Core_MPLogger.TestMPLogger.test_logger_class
    * 通过 TestSuite 指定测试 (取消文件末尾的注释并修改):

        Suite = unittest.TestSuite()
        Suite.addTest(TestMPLogger("test_logger_class"))
        Runner = unittest.TextTestRunner()
        Runner.run(Suite)

测试方法:
    test_logger_class             — 自定义 Logger 类的类型和属性
    test_init_logger              — Logger 初始化和日志级别设置
    test_main_process_logging     — 主进程日志输出
    test_child_process_logging    — 子进程日志输出
    test_child_process_with_queue — 子进程手动传递队列
    test_child_process_debug_level — 子进程 DEBUG 级别日志输出
    test_shutdown                 — Logger 关闭和资源清理
"""

import os
import sys
import logging
import subprocess
import unittest
import multiprocess

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from QuantStudio.Core.MPLogger import init_logger, logger, shutdown, get_queue, _MultiProcessLogger


# ========== 子进程任务函数 ==========

def _worker_task_simple(worker_id: int):
    """简单子进程任务"""
    logger.info(f"子进程 {worker_id} 日志")
    return worker_id


def _worker_task_with_queue(worker_id: int, queue, level):
    """带队列的子进程任务"""
    init_logger(level=level, queue=queue)
    logger.info(f"子进程 {worker_id} 手动初始化日志")
    return worker_id


def _worker_task_debug_level(worker_id: int):
    """测试子进程 DEBUG 级别日志任务"""
    from QuantStudio.Core.MPLogger import _listener
    assert _listener is None, "子进程不应启动日志监听器"
    # 验证子进程的 Logger 配置
    assert logger.level == logging.DEBUG, f"子进程 Logger 级别错误: {logger.level}"
    assert len(logger.handlers) > 0, "子进程 Logger 没有 handler"
    assert 'QueueHandler' in type(logger.handlers[0]).__name__, \
        f"子进程 Logger 使用的不是 QueueHandler: {type(logger.handlers[0])}"

    # 输出所有级别的日志
    logger.debug(f'子进程 {worker_id} DEBUG 日志')
    logger.info(f'子进程 {worker_id} INFO 日志')
    logger.warning(f'子进程 {worker_id} WARNING 日志')
    return worker_id


# ========== 测试类 ==========

class TestMPLogger(unittest.TestCase):
    """MPLogger 多进程日志模块测试"""

    @classmethod
    def setUpClass(cls):
        """保存全局日志级别和初始化状态。"""
        import QuantStudio.Core.MPLogger as MPLogger
        cls._original_level = logger.level
        cls._original_initialized = MPLogger._initialized

    @classmethod
    def tearDownClass(cls):
        """恢复全局日志状态，避免影响后续测试。"""
        shutdown()
        logger.setLevel(cls._original_level)
        if cls._original_initialized:
            init_logger(level=cls._original_level)
        else:
            for handler in logger.handlers:
                handler.setLevel(cls._original_level)

    def setUp(self):
        """每个测试前重置 Logger 状态"""
        # 关闭之前的 listener
        shutdown()

    def tearDown(self):
        """每个测试后清理"""
        shutdown()

    # ==================== 基础功能测试 ====================

    def test_logger_class(self):
        """测试自定义 Logger 类的类型和属性"""
        # 验证 logger 是自定义类的实例
        self.assertIsInstance(logger, _MultiProcessLogger, "Logger 类型错误")
        self.assertTrue(hasattr(logger, '_creation_pid'), "Logger 缺少 _creation_pid 属性")
        self.assertTrue(hasattr(logger, '_child_initialized'), "Logger 缺少 _child_initialized 属性")
        self.assertEqual(logger._creation_pid, os.getpid(), "进程号不匹配")

    def test_init_logger(self):
        """测试 Logger 初始化和日志级别设置"""
        # 初始化 Logger
        init_logger(level=logging.DEBUG)

        # 验证日志级别
        self.assertEqual(logger.level, logging.DEBUG, "日志级别错误")

        # 验证队列已创建
        queue = get_queue()
        self.assertIsNotNone(queue, "队列未创建")

    def test_main_process_logging(self):
        """测试主进程日志输出"""
        # 初始化 Logger
        init_logger(level=logging.DEBUG)

        # 输出日志（不会抛出异常即为成功）
        logger.info("主进程测试日志")
        logger.debug("主进程调试日志")

    # ==================== 子进程测试 ====================

    def test_child_process_logging(self):
        """测试子进程日志输出"""
        # 初始化 Logger
        init_logger(level=logging.DEBUG)

        # 启动子进程
        processes = []
        for i in range(2):
            p = multiprocess.Process(target=_worker_task_simple, args=(i,))
            p.start()
            processes.append(p)

        # 等待子进程完成
        for p in processes:
            p.join(timeout=10)
            self.assertEqual(p.exitcode, 0, f"子进程退出码异常: {p.exitcode}")

    def test_child_process_with_queue(self):
        """测试子进程手动传递队列"""
        # 初始化 Logger
        init_logger(level=logging.DEBUG)
        queue = get_queue()
        level = logger.level

        # 启动子进程
        processes = []
        for i in range(2):
            p = multiprocess.Process(
                target=_worker_task_with_queue,
                args=(i, queue, level)
            )
            p.start()
            processes.append(p)

        # 等待子进程完成
        for p in processes:
            p.join(timeout=10)
            self.assertEqual(p.exitcode, 0, f"子进程退出码异常: {p.exitcode}")

    def test_child_process_debug_level(self):
        """测试主进程设置 DEBUG 级别后，子进程能否输出 DEBUG 日志"""
        # 主进程设置 DEBUG 级别
        init_logger(level=logging.DEBUG)

        # 验证主进程配置
        self.assertEqual(logger.level, logging.DEBUG, "主进程 Logger 级别错误")

        # 启动子进程
        processes = []
        for i in range(2):
            p = multiprocess.Process(target=_worker_task_debug_level, args=(i,))
            p.start()
            processes.append(p)

        # 等待子进程完成
        for p in processes:
            p.join(timeout=10)
            self.assertEqual(p.exitcode, 0, f"子进程退出码异常: {p.exitcode}")

    # ==================== 资源清理测试 ====================

    def test_shutdown(self):
        """测试 Logger 关闭和资源清理"""
        # 初始化 Logger
        init_logger(level=logging.DEBUG)

        # 关闭 Logger
        shutdown()

        # 验证清理
        from QuantStudio.Core.MPLogger import _listener, _queue
        self.assertIsNone(_listener, "Listener 未清理")
        self.assertIsNone(_queue, "队列引用未清理")
        self.assertFalse(any(isinstance(h, logging.handlers.QueueHandler) for h in logger.handlers))

    def test_shutdown_then_logging_exits(self):
        """关闭后大量输出日志，解释器也应正常退出。"""
        Code = """
from QuantStudio.Core.MPLogger import init_logger, logger, shutdown
init_logger()
logger.info("关闭前日志")
shutdown()
for i in range(200):
    logger.info("关闭后日志 %s %s", i, "x" * 1000)
"""
        Result = subprocess.run(
            [sys.executable, "-c", Code],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(Result.returncode, 0, Result.stderr.decode(errors="replace"))
        self.assertEqual(Result.stderr.count(b"x" * 1000), 200)

    def test_reinitialize(self):
        """关闭后可创建新队列，重复初始化不会叠加进程包装。"""
        init_logger()
        OldQueue = get_queue()
        PatchedStart = multiprocess.Process.start
        shutdown()
        init_logger()
        self.assertIsNot(get_queue(), OldQueue)
        self.assertIs(multiprocess.Process.start, PatchedStart)
        logger.info("重新初始化后的日志")

    def test_shutdown_external_queue(self):
        """关闭日志时不关闭调用方传入的队列。"""
        Queue = multiprocess.Queue()
        try:
            init_logger(queue=Queue)
            logger.info("外部队列日志")
            shutdown()
            Queue.put("仍然可用")
            self.assertEqual(Queue.get(timeout=5), "仍然可用")
        finally:
            shutdown()
            Queue.close()
            Queue.join_thread()


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)

    # 或者通过 TestSuite 指定测试:
    # Suite = unittest.TestSuite()
    # Suite.addTest(TestMPLogger("test_child_process_debug_level"))
    # Runner = unittest.TextTestRunner(verbosity=2)
    # Runner.run(Suite)
