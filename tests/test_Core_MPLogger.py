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
        """测试类初始化"""
        # 确保使用 D:\HST\QuantStudio 目录
        if r'D:\HST\QuantStudio' not in sys.path:
            sys.path.insert(0, r'D:\HST\QuantStudio')

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
        from QuantStudio.Core.MPLogger import _listener
        self.assertIsNone(_listener, "Listener 未清理")


if __name__ == "__main__":
    # 运行所有测试
    unittest.main(verbosity=2)

    # 或者通过 TestSuite 指定测试:
    # Suite = unittest.TestSuite()
    # Suite.addTest(TestMPLogger("test_child_process_debug_level"))
    # Runner = unittest.TextTestRunner(verbosity=2)
    # Runner.run(Suite)
