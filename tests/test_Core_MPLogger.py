# -*- coding: utf-8 -*-
"""MPLogger 测试文件

测试多进程全局日志模块的功能。

运行方式:
    python -m pytest tests/test_Core_MPLogger.py -v
    或
    python tests/test_Core_MPLogger.py
"""

import os
import sys
import time
import logging
import multiprocessing

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from QuantStudio.Core.MPLogger import init_logger, logger, shutdown, get_queue, _MultiProcessLogger


def test_logger_class():
    """测试自定义 Logger 类"""
    print("测试 1: 自定义 Logger 类")

    # 验证 logger 是自定义类的实例
    assert isinstance(logger, _MultiProcessLogger), f"Logger 类型错误: {type(logger)}"
    assert hasattr(logger, '_creation_pid'), "Logger 缺少 _creation_pid 属性"
    assert hasattr(logger, '_child_initialized'), "Logger 缺少 _child_initialized 属性"
    assert logger._creation_pid == os.getpid(), "进程号不匹配"

    print("  ✓ Logger 类型正确")
    print("  ✓ 进程号记录正确")


def test_init_logger():
    """测试 Logger 初始化"""
    print("测试 2: Logger 初始化")

    # 初始化 Logger
    init_logger(level=logging.DEBUG)

    # 验证日志级别
    assert logger.level == logging.DEBUG, f"日志级别错误: {logger.level}"

    # 验证队列已创建
    queue = get_queue()
    assert queue is not None, "队列未创建"

    print("  ✓ 日志级别设置正确")
    print("  ✓ 队列创建成功")


def test_main_process_logging():
    """测试主进程日志输出"""
    print("测试 3: 主进程日志输出")

    # 初始化 Logger
    init_logger(level=logging.DEBUG)

    # 输出日志（不会抛出异常即为成功）
    logger.info("主进程测试日志")
    logger.debug("主进程调试日志")

    print("  ✓ 主进程日志输出成功")


def worker_task_simple(worker_id: int):
    """简单子进程任务"""
    logger.info(f"子进程 {worker_id} 日志")
    return worker_id


def test_child_process_logging():
    """测试子进程日志输出"""
    print("测试 4: 子进程日志输出")

    # 初始化 Logger
    init_logger(level=logging.DEBUG)

    # 启动子进程
    processes = []
    for i in range(2):
        p = multiprocessing.Process(target=worker_task_simple, args=(i,))
        p.start()
        processes.append(p)

    # 等待子进程完成
    for p in processes:
        p.join(timeout=10)
        assert p.exitcode == 0, f"子进程退出码异常: {p.exitcode}"

    print("  ✓ 子进程日志输出成功")


def worker_task_with_queue(worker_id: int, queue, level):
    """带队列的子进程任务"""
    # 手动初始化
    init_logger(level=level, queue=queue)
    logger.info(f"子进程 {worker_id} 手动初始化日志")
    return worker_id


def test_child_process_with_queue():
    """测试子进程手动传递队列"""
    print("测试 5: 子进程手动传递队列")

    # 初始化 Logger
    init_logger(level=logging.DEBUG)
    queue = get_queue()
    level = logger.level

    # 启动子进程
    processes = []
    for i in range(2):
        p = multiprocessing.Process(
            target=worker_task_with_queue,
            args=(i, queue, level)
        )
        p.start()
        processes.append(p)

    # 等待子进程完成
    for p in processes:
        p.join(timeout=10)
        assert p.exitcode == 0, f"子进程退出码异常: {p.exitcode}"

    print("  ✓ 子进程手动传递队列成功")


def test_shutdown():
    """测试 Logger 关闭"""
    print("测试 6: Logger 关闭")

    # 初始化 Logger
    init_logger(level=logging.DEBUG)

    # 关闭 Logger
    shutdown()

    # 验证清理
    from QuantStudio.Core.MPLogger import _listener
    assert _listener is None, "Listener 未清理"

    print("  ✓ Logger 关闭成功")


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("MPLogger 测试开始")
    print("=" * 50)

    try:
        test_logger_class()
        print()
        test_init_logger()
        print()
        test_main_process_logging()
        print()
        test_child_process_logging()
        print()
        test_child_process_with_queue()
        print()
        test_shutdown()

        print()
        print("=" * 50)
        print("所有测试通过 ✓")
        print("=" * 50)
        return True
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        return False
    except Exception as e:
        print(f"\n✗ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
