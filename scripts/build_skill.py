# -*- coding: utf-8 -*-
"""生成完整的 QuantStudio Skill。

Skill 由两部分组合而成：``skills/quantstudio/SKILL.md``（路由层，手工维护）
与 ``docs/*.ipynb`` 派生出的参考文档（生成物）。本脚本将二者组装到指定目录，
使 Skill 具备完整的 ``SKILL.md`` + ``references/`` 结构。

参考文档的目录结构以 ``mkdocs.yml`` 的 ``nav`` 为准，避免与文档站点漂移。
``SKILL.md`` 与 ``references/`` 始终保持平级，因此文档内部的相对链接无需重写。

用法：
    python scripts/build_skill.py --output-dir .claude/skills
    python scripts/build_skill.py --output-dir .claude/skills --clean

生成结果：
    <output_dir>/quantstudio/
        SKILL.md
        references/
            Core/*.md
            因子框架/*.md
            ...
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

import yaml

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
DOCS_SRC = ROOT / "docs"
DOCS_TOOLS = DOCS_SRC / "tools"
SKILL_SRC = ROOT / "skills" / "quantstudio"
MKDOCS_YML = ROOT / "mkdocs.yml"

# Skill 目录名
SKILL_NAME = "quantstudio"

# 不纳入 Skill 的文档（非 notebook 派生，或已失效）
EXCLUDED_DOCS = {
    "index.md",               # 站点首页，是 通则和约定.md 的副本
    "MCP/jy_doc_mcp.md",      # 手写文档，未纳入 Skill
    "回测框架/BTStorer.md",    # 手写文档，未纳入 Skill
}


def collect_reference_docs():
    """从 mkdocs.yml 的 nav 中收集需要生成的参考文档路径。

    Returns:
        set[str]: 相对 docs 目录的 markdown 路径集合，如 ``{"Core/缓存.md", ...}``
    """
    with open(MKDOCS_YML, mode="r", encoding="utf-8") as f:
        iConfig = yaml.safe_load(f)

    iDocs = set()

    def walk(nav_items):
        if isinstance(nav_items, str):
            if nav_items.endswith(".md"):
                iDocs.add(nav_items)
            return
        for iItem in nav_items:
            if isinstance(iItem, dict):
                for iValue in iItem.values():
                    walk(iValue)
            else:
                walk(iItem)

    walk(iConfig.get("nav", []))
    return iDocs - EXCLUDED_DOCS


def build_reference_docs(references_dir):
    """将 notebook 转换为 Skill 参考文档。

    先全量转换到临时目录，再按 mkdocs nav 挑选，使输出保持 nav 的目录组织。

    Args:
        references_dir: 参考文档输出目录

    Returns:
        list[str]: 未找到 notebook 源的文档路径列表
    """
    sys.path.insert(0, str(DOCS_TOOLS))
    from gen_doc import main as gen_doc_main

    iWanted = collect_reference_docs()
    print(f"  mkdocs nav 中的参考文档: {len(iWanted)} 篇")

    iTmpDir = references_dir.parent / "_references_tmp"
    if iTmpDir.exists():
        shutil.rmtree(iTmpDir)
    gen_doc_main(
        target_dir=str(iTmpDir),
        source_dir=str(DOCS_SRC),
        skill_mode=True,
        strip_images=True,
        drop_exec_count=True,
    )

    iMissing = []
    references_dir.mkdir(parents=True, exist_ok=True)
    for iRelDoc in sorted(iWanted):
        iSrc = iTmpDir / iRelDoc
        if not iSrc.exists():
            print(f"  [跳过] 未找到: {iRelDoc}")
            iMissing.append(iRelDoc)
            continue
        iDst = references_dir / iRelDoc
        iDst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(iSrc), str(iDst))
    shutil.rmtree(iTmpDir, ignore_errors=True)
    return iMissing


def copy_skill_md(target_skill_dir):
    """拷贝手工维护的 SKILL.md。"""
    iSource = SKILL_SRC / "SKILL.md"
    if not iSource.exists():
        raise FileNotFoundError(f"未找到 SKILL.md: {iSource}")
    shutil.copy2(iSource, target_skill_dir / "SKILL.md")


def clean(output_dir):
    """删除生成的 Skill 目录。"""
    iTarget = Path(output_dir) / SKILL_NAME
    if iTarget.exists():
        shutil.rmtree(iTarget)
        print(f"已删除: {iTarget}")


def main():
    parser = argparse.ArgumentParser(description="生成完整的 QuantStudio Skill")
    parser.add_argument("--output-dir", required=True, help="Skill 生成的目标目录")
    parser.add_argument("--clean", action="store_true", help="删除已生成的 Skill")
    args = parser.parse_args()

    iOutputDir = Path(args.output_dir).expanduser().resolve()

    if args.clean:
        clean(iOutputDir)
        return

    iTargetSkillDir = iOutputDir / SKILL_NAME
    iReferencesDir = iTargetSkillDir / "references"

    print("=" * 50)
    print(f"生成 Skill: {iTargetSkillDir}")
    print("=" * 50)

    iTargetSkillDir.mkdir(parents=True, exist_ok=True)

    print()
    print("步骤 1/2：转换 notebook → 参考文档")
    iFailed = build_reference_docs(iReferencesDir)

    print()
    print("步骤 2/2：拷贝 SKILL.md")
    copy_skill_md(iTargetSkillDir)
    print(f"  {SKILL_SRC / 'SKILL.md'} → {iTargetSkillDir / 'SKILL.md'}")

    print()
    print("=" * 50)
    print("完成！")
    print(f"  Skill 目录: {iTargetSkillDir}")
    if iFailed:
        print(f"  未生成的文档 ({len(iFailed)} 篇): {', '.join(iFailed)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
