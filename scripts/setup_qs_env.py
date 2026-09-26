#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""QuantStudio 环境配置脚本。

把 QuantStudio 源码仓库中的模板与 Skill 配置到一个目标目录下，产出一份可直接
使用的开发环境。所有内容均取自脚本所在的 QuantStudio 仓库（源码模式），
不需要预先安装 QuantStudio 包。

写入目标目录（`--target-dir`，默认为本仓库根目录）的内容：

    .mcp.json                     MCP 服务配置
    CLAUDE.local.md               本地约定（含 Python 解释器路径）
    .claude/skills/*              skills 目录下各 SKILL 的链接
    .claude/skills/quantstudio/   文档派生型 Skill（由 build_skill.py 生成，非链接）

使用方法:
    # 配置本仓库自身（默认：--target-dir 为脚本所在仓库根）
    python scripts/setup_qs_env.py

    # 把 QS 环境配置到指定目录
    python scripts/setup_qs_env.py --target-dir D:/Project/MyProject

    # 跳过依赖安装（依赖已就绪时）
    python scripts/setup_qs_env.py --skip-pip

    # 指定 MCP 文档缓存目录（默认 D:/Data/JYDBDoc）
    python scripts/setup_qs_env.py --cache-dir E:/Cache/JYDBDoc

    # 覆盖已存在的 .mcp.json / CLAUDE.local.md / skill 链接与生成产物
    python scripts/setup_qs_env.py --force

    # 仅预览将要执行的操作，不做任何改动
    python scripts/setup_qs_env.py --dry-run

    # 指定 pip 使用的镜像源
    python scripts/setup_qs_env.py --pip-index-url https://pypi.tuna.tsinghua.edu.cn/simple

前置要求:
    - 建议使用 QS 环境的解释器执行本脚本（依赖安装的目标即该解释器）；
      若解释器路径不是虚拟环境（无 pyvenv.cfg），脚本会提示。
    - 生成 skill 链接时，Windows 下符号链接需开发者模式或管理员权限；
      权限不足时会自动回退为目录联接。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_SRC = REPO_ROOT / "skills"
TEMPLATE_MCP = REPO_ROOT / ".mcp.example.json"
TEMPLATE_CLAUDE = REPO_ROOT / "CLAUDE.local.example.md"
BUILD_SKILL = REPO_ROOT / "scripts" / "build_skill.py"
DEFAULT_CACHE_DIR = "D:/Data/JYDBDoc"

# 文档派生型 Skill：其 references/ 由 docs/ 生成（见 CLAUDE.md「Skill 维护」），
# 因此不能链接——链接只会得到缺少 references/ 的空壳，必须用 build_skill.py 生成。
GENERATED_SKILLS = {"quantstudio"}


def _log(msg: str, *, level: str = "INFO") -> None:
    prefix = {"INFO": "[INFO]", "WARN": "[WARN]", "SKIP": "[SKIP]", "DRY": "[DRY-RUN]"}[level]
    print(f"{prefix} {msg}")


def _resolve_dir(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def install_requirements(python: str, dry_run: bool, index_url: str | None) -> None:
    """在当前解释器中安装本仓库 requirements.txt 中的依赖。"""
    req = REPO_ROOT / "requirements.txt"
    if not req.exists():
        raise FileNotFoundError(f"未找到依赖清单: {req}")

    cmd = [python, "-m", "pip", "install", "-r", str(req)]
    if index_url:
        cmd += ["-i", index_url]

    if dry_run:
        _log(f"将执行: {' '.join(cmd)}", level="DRY")
        return

    _log(f"安装依赖: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def write_mcp_config(target_dir: Path, python: str, cache_dir: str, force: bool, dry_run: bool) -> None:
    """渲染 .mcp.example.json 生成目标目录下的 .mcp.json。"""
    target = target_dir / ".mcp.json"
    if target.exists() and not force:
        _log(f"{target} 已存在，跳过（--force 可覆盖）", level="SKIP")
        return

    text = TEMPLATE_MCP.read_text(encoding="utf-8")
    replacements = {
        "{{PYTHON}}": python,
        "{{SERVER_PY}}": (REPO_ROOT / "mcp" / "jy_doc" / "server.py").as_posix(),
        "{{PROJECT}}": target_dir.as_posix(),
        "{{CACHE_DIR}}": _resolve_dir(cache_dir).as_posix(),
        "{{PYTHONPATH}}": REPO_ROOT.as_posix(),
    }
    for key, value in replacements.items():
        text = text.replace(key, value.replace("\\", "\\\\"))

    # 校验替换后仍是合法 JSON；残留占位符会让解析失败或漏改字段
    if "{{" in text:
        leftover = text[text.index("{{") : text.index("{{") + 40]
        raise ValueError(f"模板中存在未替换的占位符: {leftover}")
    json.loads(text)

    if dry_run:
        _log(f"将写入 {target}", level="DRY")
        print(text)
        return

    target.write_text(text, encoding="utf-8")
    _log(f"已生成 {target}")


def _create_link(src: Path, dst: Path) -> str:
    """建立目录链接，返回实际使用的方式。"""
    try:
        dst.symlink_to(src, target_is_directory=True)
        return "symlink"
    except OSError:
        # 符号链接不可用（Windows 未开启开发者模式），回退到目录联接
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise OSError(f"symlink 与 junction 均创建失败: {result.stdout}{result.stderr}")
        return "junction"


def _is_link_to(path: Path, target: Path) -> bool:
    """判断 path 是否为指向 target 的链接（symlink 或 Windows junction）。"""
    if not os.path.islink(path) and not path.exists():
        return False
    return os.path.realpath(path) == os.path.realpath(target)


def link_skills(target_dir: Path, force: bool, dry_run: bool) -> None:
    """把本仓库 skills/ 下的每个 SKILL 链接到目标目录的 .claude/skills/。

    文档派生型 Skill（见 GENERATED_SKILLS）不在此处理，由 build_generated_skills 生成。
    """
    if not SKILLS_SRC.is_dir():
        raise FileNotFoundError(f"未找到 skill 源目录: {SKILLS_SRC}")

    sources = sorted(
        p for p in SKILLS_SRC.iterdir()
        if (p / "SKILL.md").is_file() and p.name not in GENERATED_SKILLS
    )
    if not sources:
        _log(f"{SKILLS_SRC} 下没有找到含 SKILL.md 的目录", level="WARN")
        return

    dst_root = target_dir / ".claude" / "skills"
    if not dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    for src in sources:
        dst = dst_root / src.name
        # 已指向同一目标的链接视为已配置
        if _is_link_to(dst, src):
            _log(f"{dst.name} 链接已存在，跳过", level="SKIP")
            continue

        if dst.exists() or os.path.islink(dst):
            if not force:
                _log(f"{dst} 已存在且非本脚本创建的链接，跳过（--force 可覆盖）", level="SKIP")
                continue
            if dry_run:
                _log(f"将删除并重建链接: {dst}", level="DRY")
            elif os.path.islink(dst):
                dst.unlink()
            else:
                _log(f"{dst} 是真实目录而非链接，拒绝删除；请手动处理", level="WARN")
                continue

        if dry_run:
            _log(f"将创建链接: {dst} -> {src}", level="DRY")
            continue

        kind = _create_link(src, dst)
        _log(f"已创建链接({kind}): {dst.name} -> {src}")


def build_generated_skills(target_dir: Path, force: bool, dry_run: bool) -> None:
    """用 build_skill.py 生成文档派生型 Skill 到目标目录的 .claude/skills/。

    内容取自本仓库的 docs/ 与 mkdocs.yml（源码模式），与 --target-dir 指向何处无关。
    """
    if not GENERATED_SKILLS:
        return
    if not BUILD_SKILL.is_file():
        raise FileNotFoundError(f"未找到 Skill 生成脚本: {BUILD_SKILL}")

    dst_root = target_dir / ".claude" / "skills"

    for name in sorted(GENERATED_SKILLS):
        dst = dst_root / name
        cmd = [sys.executable, str(BUILD_SKILL), "--output-dir", str(dst_root)]

        if dry_run:
            if dst.exists() and force:
                _log(f"将删除后重新生成 Skill: {dst}", level="DRY")
            _log(f"将生成 Skill: {' '.join(cmd)}", level="DRY")
            continue

        if dst.exists():
            if not force:
                _log(f"{dst} 已存在，跳过（--force 可删除重建）", level="SKIP")
                continue
            # build_skill.py 的 --clean 是终止性的（清理后即退出），故在此先行删除
            shutil.rmtree(dst)
            _log(f"已删除旧产物: {dst}")

        _log(f"生成 Skill {name}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def write_claude_local(target_dir: Path, python: str, force: bool, dry_run: bool) -> None:
    """渲染 CLAUDE.local.example.md 生成目标目录下的 CLAUDE.local.md。"""
    target = target_dir / "CLAUDE.local.md"
    if target.exists() and not force:
        _log(f"{target} 已存在，跳过（--force 可覆盖）", level="SKIP")
        return

    text = TEMPLATE_CLAUDE.read_text(encoding="utf-8")
    lines = []
    replaced = False
    for line in text.splitlines(keepends=True):
        if not replaced and line.lstrip().startswith("Python："):
            lines.append(f"Python：使用 QS 环境，解释器位置是：{python}\n")
            replaced = True
        else:
            lines.append(line)

    if not replaced:
        _log("模板中未匹配到 Python 路径行，原样复制", level="WARN")
        lines = [text]

    rendered = "".join(lines)

    if dry_run:
        _log(f"将写入 {target}", level="DRY")
        print(rendered)
        return

    target.write_text(rendered, encoding="utf-8")
    _log(f"已生成 {target}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把 QuantStudio 的模板与 Skill 配置到目标目录（安装依赖、生成 MCP 配置、链接 Skill、生成本地约定文件）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help="配置写入的目标目录；未指定时使用本仓库根目录",
    )
    parser.add_argument("--skip-pip", action="store_true", help="跳过依赖安装")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help=f"聚源文档缓存目录 (默认: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的 .mcp.json / CLAUDE.local.md / skill 链接与生成产物")
    parser.add_argument("--dry-run", action="store_true", help="仅预览操作，不做任何改动")
    parser.add_argument("--pip-index-url", default=None, help="pip 镜像源地址")
    args = parser.parse_args()

    python = sys.executable
    if not python:
        _log("无法获取当前解释器路径 (sys.executable 为空)", level="WARN")
        return 1
    # 不能对解释器路径做 resolve()：venv 的 bin/python 是指向基础解释器的符号链接，
    # 解析后会丢失虚拟环境身份，导致依赖装到基础解释器里
    python = str(Path(python).absolute())

    if args.target_dir:
        target_dir = _resolve_dir(args.target_dir)
        if not target_dir.is_dir():
            raise NotADirectoryError(f"--target-dir 指定的目录不存在: {target_dir}")
    else:
        target_dir = REPO_ROOT
        _log(f"未指定 --target-dir，使用本仓库根目录: {target_dir}")

    print(f"仓库根目录: {REPO_ROOT}")
    print(f"配置目标目录: {target_dir}")
    print(f"Python 解释器: {python}")
    print(f"MCP 缓存目录: {_resolve_dir(args.cache_dir)}")
    print("-" * 60)

    if os.name == "nt" and not (Path(python).parent.parent / "pyvenv.cfg").exists():
        _log(f"{python} 看起来不是虚拟环境，依赖将装到该解释器的 site-packages", level="WARN")

    if args.skip_pip:
        _log("按参数跳过依赖安装", level="SKIP")
    else:
        install_requirements(python, args.dry_run, args.pip_index_url)

    write_mcp_config(target_dir, python, args.cache_dir, args.force, args.dry_run)
    link_skills(target_dir, args.force, args.dry_run)
    build_generated_skills(target_dir, args.force, args.dry_run)
    write_claude_local(target_dir, python, args.force, args.dry_run)

    print("-" * 60)
    print("配置完成。" + ("（dry-run，未做任何改动）" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
