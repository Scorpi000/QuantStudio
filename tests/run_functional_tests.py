# -*- coding: utf-8 -*-
"""运行功能回归，排除没有断言的文件格式性能基准。

实现逻辑：发现 tests 下的 unittest 用例，递归展开测试套件，仅排除
TestFileIOBenchmark。以真实文件作为主入口，使 Windows 多进程测试可以
重新加载入口；结果包含各测试名称及失败详情。

使用方法：在仓库根目录运行 ``python tests/run_functional_tests.py``。
可通过 ``--pattern`` 选择文件，例如 ``--pattern test_Core_*.py``，
通过 ``--log`` 将测试报告保存到指定文件。
通过重复的 ``--exclude`` 通配模式显式跳过外部服务或单独运行的耗时用例。
"""
import argparse
from fnmatch import fnmatchcase
import logging
from pathlib import Path
import sys
import unittest


def iter_cases(suite, excludes=()):
    """递归展开测试套件并过滤性能基准。

    Args:
        suite: unittest 测试套件。
        excludes: 测试完整名称的排除模式。

    Yields:
        包含断言或功能验证的测试用例。
    """
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_cases(item, excludes)
        elif type(item).__name__ != "TestFileIOBenchmark":
            if any(fnmatchcase(item.id(), pattern) for pattern in excludes):
                class_name = f"{type(item).__module__}.{type(item).__qualname__}.__class__"
                if any(fnmatchcase(class_name, pattern) for pattern in excludes):
                    type(item).__unittest_skip__ = True
                    type(item).__unittest_skip_why__ = "按显式测试范围排除"
                @unittest.skip("按显式测试范围排除")
                def excluded():
                    pass
                setattr(item, item._testMethodName, excluded)
            yield item


def main():
    """解析测试范围并返回功能回归退出码。

    Returns:
        全部通过时返回零，失败或错误时返回一。
    """
    parser = argparse.ArgumentParser(description="运行 QuantStudio 功能回归")
    parser.add_argument("--pattern", default="test_*.py", help="测试文件匹配模式")
    parser.add_argument("--log", help="UTF-8 测试报告路径")
    parser.add_argument("--exclude", action="append", default=[], help="显式排除测试名称的通配模式")
    args = parser.parse_args()
    from QuantStudio.Core import setDefaultLogLevel
    setDefaultLogLevel(logging.CRITICAL)
    suite = unittest.TestSuite(iter_cases(unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern=args.pattern), args.exclude))
    stream = open(args.log, "w", encoding="utf-8", buffering=1) if args.log else sys.stderr
    try:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    finally:
        if args.log:
            stream.close()
    print(f"测试 {result.testsRun}，失败 {len(result.failures)}，错误 {len(result.errors)}，跳过 {len(result.skipped)}", flush=True)
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    raise SystemExit(main())
