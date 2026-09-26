#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""聚源数据库文档索引预抓取脚本。

本脚本登录聚源数据字典平台(dd.gildata.com)，遍历所有数据库的目录树结构，
构建本地搜索索引，供 mcp/jy_doc MCP 服务使用。

使用方法:
    # 使用默认缓存目录和环境变量凭据
    python scripts/scrape_jy_doc.py

    # 指定缓存目录
    python scripts/scrape_jy_doc.py --cache-dir E:/Cache/JYDBDoc

    # 指定登录凭据
    python scripts/scrape_jy_doc.py --user xxx --pwd xxx

    # 通过环境变量指定（也可写在仓库根 .env 中）
    set JY_DOC_CACHE=D:\\Data\\JYDBDoc
    set JY_DOC_USER=xxx
    set JY_DOC_PWD=xxx
    python scripts/scrape_jy_doc.py

生成文件:
    <cache_dir>/tree_index.json  — 包含所有数据库和表的索引文件

前置要求:
    - ddddocr (pip install ddddocr) — 用于验证码识别
    - 聚源数据字典平台的登录账号
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 仓库根目录与聚源文档配置目录
_ROOT = Path(__file__).resolve().parent.parent
_MCP_DIR = _ROOT / "mcp"


def _load_scraper():
    """按路径加载 mcp/jy_doc/scraper.py。

    顶层 `mcp` 这个名字已被已安装的 MCP SDK 占用（`import mcp` 会命中 SDK 而非
    本仓库的 mcp/ 目录），因此不能写 `from mcp.jy_doc.scraper import ...`。
    这里用 importlib 把 jy_doc 包按路径注册为 `jy_doc`，使 scraper 内部的
    相对导入（`from .fetcher import ...`）正常生效。
    """
    pkg_dir = _MCP_DIR / "jy_doc"

    def _load(name, filename, is_pkg=False):
        kwargs = {"submodule_search_locations": [str(pkg_dir)]} if is_pkg else {}
        spec = importlib.util.spec_from_file_location(
            name, str(pkg_dir / filename), **kwargs
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    _load("jy_doc", "__init__.py", is_pkg=True)
    _load("jy_doc.models", "models.py")
    return _load("jy_doc.scraper", "scraper.py")


scrape_and_save = _load_scraper().scrape_and_save


def main():
    # 加载 mcp/.env，使 JY_DOC_* 环境变量对 argparse 默认值生效
    load_dotenv(_MCP_DIR / ".env")

    parser = argparse.ArgumentParser(
        description="聚源数据库文档索引预抓取脚本 — 构建本地搜索索引"
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("JY_DOC_CACHE", r"D:\Data\JYDBDoc"),
        help="本地缓存目录路径 (默认: D:\\Data\\JYDBDoc 或 JY_DOC_CACHE 环境变量)",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("JY_DOC_USER", ""),
        help="聚源文档平台用户名 (默认: JY_DOC_USER 环境变量)",
    )
    parser.add_argument(
        "--pwd",
        default=os.getenv("JY_DOC_PWD", ""),
        help="聚源文档平台密码 (默认: JY_DOC_PWD 环境变量)",
    )
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # 检查凭据
    user = args.user or os.getenv("JY_DOC_USER", "")
    pwd = args.pwd or os.getenv("JY_DOC_PWD", "")
    if not user or not pwd:
        print("错误: 未配置聚源文档平台登录凭据")
        print("请通过 --user/--pwd 参数或 JY_DOC_USER/JY_DOC_PWD 环境变量指定")
        sys.exit(1)

    # 检查 ddddocr
    try:
        import ddddocr
        print(f"ddddocr 已安装")
    except ImportError:
        print("错误: 需要安装 ddddocr 库以识别验证码")
        print("请运行: pip install ddddocr")
        sys.exit(1)

    print(f"缓存目录: {args.cache_dir}")
    print(f"登录用户: {user}")
    print("-" * 50)

    # 执行抓取
    try:
        index_data = scrape_and_save(
            cache_dir=args.cache_dir,
            user=user,
            pwd=pwd,
        )
    except RuntimeError as e:
        print(f"\n抓取失败: {e}")
        sys.exit(1)

    # 打印统计信息
    total_tables = len(index_data.get("flat_index", []))
    db_count = len(index_data.get("databases", []))

    print("-" * 50)
    print(f"抓取完成!")
    print(f"  数据库数: {db_count}")
    print(f"  表总数: {total_tables}")
    print(f"  索引文件: {os.path.join(args.cache_dir, 'tree_index.json')}")
    print(f"\n启动 MCP 服务:")
    print(f"  python mcp/jy_doc/server.py --cache-dir {args.cache_dir}")


if __name__ == "__main__":
    main()
