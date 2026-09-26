#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""QuantStudio Docker 镜像构建脚本。

构建一个自带 QuantStudio 运行环境的镜像：镜像内创建 QS 虚拟环境、clone
QuantStudio 源码，并以 ~/workspace 为目标目录完成 QS 环境配置。容器内目录约定：

    ~/python_envs/QS    QS 虚拟环境
    ~/packages/QuantStudio  QuantStudio 源码（git clone）
    ~/workspace         QS 环境配置目标目录（.mcp.json / CLAUDE.local.md / .claude/skills）
    ~/data              挂载点：JYDBDoc 文档缓存、HDF5DB 因子库

构建所需的 Dockerfile 位于 docker/Dockerfile，构建上下文为仓库根目录
（需将 requirements.txt、skills/ 等一并送入构建）。

使用方法:
    # 使用默认镜像名与标签构建（从远端 clone 源码）
    python scripts/build_docker.py

    # 指定镜像名与标签
    python scripts/build_docker.py --tag qs-env:dev

    # 使用本地构建上下文中的源码（不依赖远端分支，便于测试）
    python scripts/build_docker.py --source copy

    # 指定要 clone 的仓库与分支
    python scripts/build_docker.py --repo-url https://github.com/Scorpi000/QuantStudio.git --branch v2

    # 构建后列出镜像信息
    python scripts/build_docker.py --inspect

    # 使用指定的 Dockerfile / 构建上下文
    python scripts/build_docker.py --dockerfile docker/Dockerfile --context .

    # 仅打印将执行的 docker 命令，不实际构建
    python scripts/build_docker.py --dry-run

    # 构建后运行容器打印自检信息（进入 QS 环境并 import QuantStudio）
    python scripts/build_docker.py --smoke-test --source copy

    # 打印运行容器的挂载命令（JYDBDoc / HDF5DB 数据目录）
    python scripts/build_docker.py --run-cmd --data-dir D:/Data

前置要求:
    - 本机已安装 Docker 且 Docker daemon 正在运行。
    - 从远端 clone 时需联网（apt 安装、pip 安装、git clone）。
    - `--source copy`（默认）使用本地构建上下文，不依赖远端分支，适合测试。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
DEFAULT_TAG = "quantstudio:latest"
DEFAULT_REPO_URL = "https://github.com/Scorpi000/QuantStudio.git"
DEFAULT_BRANCH = "v2"
DEFAULT_SOURCE = "copy"


def _log(msg: str, *, level: str = "INFO") -> None:
    prefix = {"INFO": "[INFO]", "WARN": "[WARN]", "DRY": "[DRY-RUN]"}[level]
    print(f"{prefix} {msg}")


def _run(cmd: list[str], dry_run: bool) -> int:
    if dry_run:
        _log(f"将执行: {' '.join(cmd)}", level="DRY")
        return 0
    _log(f"执行: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def check_docker() -> bool:
    """确认 docker CLI 可用且 daemon 正在运行。"""
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.OSType}}/{{.Architecture}}"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        _log("未找到 docker 命令，请先安装 Docker", level="WARN")
        return False

    if result.returncode != 0:
        _log(f"docker daemon 不可用: {result.stderr.strip()}", level="WARN")
        return False

    _log(f"Docker 可用: {result.stdout.strip()}")
    return True


def build_image(dockerfile: Path, context: Path, tag: str, build_args: list[str], dry_run: bool) -> int:
    if not dockerfile.exists():
        _log(f"未找到 Dockerfile: {dockerfile}", level="WARN")
        return 1
    if not context.is_dir():
        _log(f"构建上下文目录不存在: {context}", level="WARN")
        return 1

    cmd = ["docker", "build", "-f", str(dockerfile), "-t", tag]
    for arg in build_args:
        cmd += ["--build-arg", arg]
    cmd.append(str(context))

    return _run(cmd, dry_run)


def print_run_cmd(tag: str, data_dir: str | None) -> None:
    """打印运行容器的命令，含 JYDBDoc / HDF5DB 数据目录挂载。"""
    cmd = ["docker", "run", "--rm", "-it"]
    if data_dir:
        cmd += ["-v", f"{data_dir}:/root/data"]
    else:
        cmd += ["-v", "<宿主机数据目录>:/root/data"]
    cmd += ["-v", "<宿主机配置目录>:/root/.QuantStudioConfig"]
    cmd += [tag, "bash"]

    print("运行容器（按需调整挂载路径）:")
    print("  " + " ".join(cmd))
    print()
    print("说明:")
    print("  /root/data               挂载后其下的 JYDBDoc、HDF5DB 即可被容器访问")
    print("  /root/data/JYDBDoc       聚源文档缓存（setup_qs_env.py --cache-dir 的写入位置）")
    print("  /root/data/HDF5DB        HDF5 因子库")
    print("  /root/.QuantStudioConfig JYDB 等数据库连接配置")


def inspect_image(tag: str, dry_run: bool) -> int:
    return _run(["docker", "images", tag], dry_run)


def smoke_test(tag: str, dry_run: bool) -> int:
    """在容器内校验 QS 虚拟环境、源码与环境配置均已就绪。"""
    check = (
        'set -e; '
        'echo "--- python ---"; "${QS_VENV}/bin/python" -V; '
        'echo "--- import QuantStudio ---"; '
        '"${QS_VENV}/bin/python" -c "import QuantStudio; print(QuantStudio.__file__)"; '
        'echo "--- workspace ---"; ls -a "${QS_WORKSPACE}"; '
        'echo "--- skills ---"; ls "${QS_WORKSPACE}/.claude/skills"'
    )
    return _run(["docker", "run", "--rm", tag, "bash", "-c", check], dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="构建 QuantStudio Docker 镜像",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tag", default=DEFAULT_TAG, help=f"镜像名:标签 (默认: {DEFAULT_TAG})")
    parser.add_argument("--dockerfile", default=str(DEFAULT_DOCKERFILE), help="Dockerfile 路径")
    parser.add_argument("--context", default=str(REPO_ROOT), help="构建上下文目录 (默认: 仓库根)")
    parser.add_argument(
        "--source",
        choices=["copy", "git"],
        default=DEFAULT_SOURCE,
        help=f"源码来源: copy=构建上下文副本(默认, 便于测试), git=从远端 clone",
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help=f"git clone 的仓库地址 (默认: {DEFAULT_REPO_URL})")
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help=f"git clone 的分支 (默认: {DEFAULT_BRANCH})")
    parser.add_argument("--build-arg", action="append", default=[], metavar="K=V", help="额外传给 docker build 的参数，可重复")
    parser.add_argument("--data-dir", default=None, help="宿主机数据目录，用于 --run-cmd 拼接挂载命令")
    parser.add_argument("--run-cmd", action="store_true", help="打印运行容器的挂载命令（不构建）")
    parser.add_argument("--inspect", action="store_true", help="构建后列出镜像信息")
    parser.add_argument("--smoke-test", action="store_true", help="构建后运行自检容器")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不实际构建")
    args = parser.parse_args()

    dockerfile = Path(args.dockerfile).resolve()
    context = Path(args.context).resolve()

    if args.run_cmd:
        print_run_cmd(args.tag, args.data_dir)
        return 0

    print(f"仓库根目录: {REPO_ROOT}")
    print(f"Dockerfile: {dockerfile}")
    print(f"构建上下文: {context}")
    print(f"镜像标签: {args.tag}")
    print(f"源码来源: {args.source}" + (
        f" (副本自 {context})" if args.source == "copy" else f" (clone {args.repo_url} @ {args.branch})"
    ))
    print("-" * 60)

    if not args.dry_run and not check_docker():
        return 1

    build_args = [
        f"QS_SOURCE={args.source}",
        f"QS_GIT_URL={args.repo_url}",
        f"QS_GIT_BRANCH={args.branch}",
    ] + args.build_arg

    code = build_image(dockerfile, context, args.tag, build_args, args.dry_run)
    if code != 0:
        _log("构建失败", level="WARN")
        return code

    if args.inspect:
        inspect_image(args.tag, args.dry_run)

    if args.smoke_test:
        code = smoke_test(args.tag, args.dry_run)
        if code != 0:
            _log("自检失败", level="WARN")
            return code

    print("-" * 60)
    print(f"完成。镜像: {args.tag}")
    print(f"进入容器: docker run --rm -it {args.tag}")
    print(f"查看挂载命令: python {Path(__file__).name} --run-cmd")
    return 0


if __name__ == "__main__":
    sys.exit(main())
