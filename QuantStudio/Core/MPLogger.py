# -*- coding: utf-8 -*-
"""多进程全局日志模块

基于标准库 QueueHandler + QueueListener 实现，支持多进程场景下的日志统一输出。

特性:
    - 完全无感：下游代码无需任何修改
    - 自动检测：通过进程号自动判断是否在子进程中
    - 自动初始化：子进程中首次输出日志时自动配置 QueueHandler
    - 跨平台：支持 fork/spawn/forkserver 所有启动方式

使用方式:
    # 主进程中初始化（可选，用于设置日志级别）
    from QuantStudio.Core.MPLogger import init_logger
    init_logger(level=logging.DEBUG)

    # 任何地方使用（主进程/子进程都一样，无需额外配置）
    from QuantStudio.Core.MPLogger import logger
    logger.debug("这条日志会统一在主进程输出")

架构:
    主进程: Logger -> QueueHandler -> Queue -> QueueListener -> Console
    子进程: Logger -> QueueHandler -> Queue (发送到主进程)
"""

import os
import sys
import atexit
import logging
import logging.handlers
import multiprocessing
from typing import Optional, Union

__all__ = ["logger", "init_logger", "shutdown", "get_queue"]


# ========== 全局状态 ==========

# 全局队列（模块级变量，fork 模式下子进程会自动继承）
_queue: Optional[multiprocessing.Queue] = None
_listener: Optional[logging.handlers.QueueListener] = None
_main_pid: int = os.getpid()
_initialized: bool = False

# 用于 spawn 模式下传递队列和级别的模块级变量
_child_queue: Optional[multiprocessing.Queue] = None
_child_level: int = logging.INFO


class _MultiProcessLogger(logging.Logger):
    """多进程感知的 Logger 类

    在日志被实际输出时，自动检测进程号变化并初始化
    """

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        # 记录创建时的进程号
        self._creation_pid = os.getpid()
        # 子进程是否已初始化
        self._child_initialized = False

    def _check_and_init_for_child(self):
        """检查是否在子进程中，如果是则自动初始化"""
        if self._child_initialized:
            return

        current_pid = os.getpid()
        if current_pid != self._creation_pid:
            # 检测到进程号变化，说明在子进程中
            self._child_initialized = True

            # 清除已有的 handler（子进程可能继承了父进程的 handler）
            for h in self.handlers[:]:
                self.removeHandler(h)

            # 如果有全局队列，添加 QueueHandler
            if _queue is not None:
                queue_handler = logging.handlers.QueueHandler(_queue)
                self.addHandler(queue_handler)
            else:
                # 没有队列，使用标准 StreamHandler（降级方案）
                handler = logging.StreamHandler()
                handler.setLevel(self.level)
                handler.setFormatter(
                    logging.Formatter("%(asctime)s | %(name)s | %(process)d | %(levelname)s : %(message)s")
                )
                self.addHandler(handler)

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        # 在输出日志前检查并初始化
        self._check_and_init_for_child()
        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)


# 注册自定义 Logger 类
logging.setLoggerClass(_MultiProcessLogger)


def _create_handler(level: int) -> logging.StreamHandler:
    """创建标准输出 handler"""
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(process)d | %(levelname)s : %(message)s")
    )
    return handler


def _ensure_queue():
    """确保队列已创建"""
    global _queue
    if _queue is None:
        _queue = multiprocessing.Queue()
    return _queue


def _ensure_listener(level: int = logging.INFO):
    """确保 QueueListener 已启动"""
    global _listener, _initialized

    if _listener is not None:
        return

    queue = _ensure_queue()

    # 创建 QueueListener，将日志输出到控制台
    console_handler = _create_handler(level)
    _listener = logging.handlers.QueueListener(
        queue,
        console_handler,
        respect_handler_level=True
    )
    _listener.start()
    _initialized = True

    # 注册退出清理
    atexit.register(shutdown)


def _wrapped_target(original_target, queue, level, args, kwargs):
    """包装的 target 函数，在子进程中自动初始化 Logger

    这是模块级函数，可以被 pickle
    """
    # 子进程中初始化 Logger
    init_logger(level=level, queue=queue)
    # 调用原始 target
    return original_target(*args, **kwargs)


def _patch_process_start():
    """Monkey-patch multiprocessing.Process.start 方法

    在子进程启动前，自动将日志队列注入到进程参数中
    """
    original_start = multiprocessing.Process.start

    def patched_start(self, *args, **kwargs):
        # 获取当前的队列和日志级别
        queue = get_queue()
        level = logger.level

        # 将队列和级别存储在进程对象中
        if queue is not None:
            # 保存原始的 target 函数和参数
            original_target = self._target
            original_args = self._args
            original_kwargs = self._kwargs or {}

            # 使用模块级包装函数
            self._target = _wrapped_target
            self._args = (original_target, queue, level, original_args, original_kwargs)
            self._kwargs = {}

        return original_start(self, *args, **kwargs)

    multiprocessing.Process.start = patched_start


# ========== 公共 API ==========

# 全局 Logger 实例（使用自定义类）
logging.setLoggerClass(_MultiProcessLogger)
logger: _MultiProcessLogger = logging.getLogger("QS")  # type: ignore
# 强制修改 logger 的类（如果它不是自定义类）
if not isinstance(logger, _MultiProcessLogger):
    logger.__class__ = _MultiProcessLogger
    logger._creation_pid = os.getpid()
    logger._child_initialized = False


def init_logger(
    level: int = logging.INFO,
    queue: Optional[multiprocessing.Queue] = None
) -> None:
    """初始化多进程日志系统

    Args:
        level: 日志级别，默认 INFO
        queue: 外部传入的队列，如果为 None 则自动创建
    """
    global _queue, _main_pid, _initialized

    logger.setLevel(level)

    current_pid = os.getpid()
    if current_pid == _main_pid:
        # 主进程
        _queue = queue or _ensure_queue()
        _ensure_listener(level)

        # 清除已有的 handler
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        # 添加 QueueHandler
        queue_handler = logging.handlers.QueueHandler(_queue)
        logger.addHandler(queue_handler)

        # Patch Process.start 以自动传递队列
        _patch_process_start()
    else:
        # 子进程（用户显式调用或自动调用）
        _queue = queue if queue is not None else _queue
        if _queue is not None:
            # 清除已有的 handler
            for h in logger.handlers[:]:
                logger.removeHandler(h)
            # 添加 QueueHandler
            queue_handler = logging.handlers.QueueHandler(_queue)
            logger.addHandler(queue_handler)
        logger._child_initialized = True


def get_queue() -> Optional[multiprocessing.Queue]:
    """获取日志队列（用于传递给子进程）"""
    _ensure_queue()
    return _queue


def shutdown() -> None:
    """关闭日志系统"""
    global _listener, _initialized
    if _listener is not None:
        _listener.stop()
        _listener = None
    _initialized = False


# ========== 模块初始化 ==========

# 在主进程中自动创建队列
if os.getpid() == _main_pid:
    _ensure_queue()
