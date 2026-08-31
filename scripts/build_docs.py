"""一键构建 QuantStudio 文档站点。

流程：notebook → markdown（gen_doc） → mkdocs build → 静态站点

用法：
    python scripts/build_docs.py          # 构建到 site/ 目录
    python scripts/build_docs.py --serve  # 本地预览（http://127.0.0.1:8000）
    python scripts/build_docs.py --clean  # 清理生成的文件
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
DOCS_SRC = ROOT / "docs"
DOCS_MD = ROOT / "docs_md"
SITE_DIR = ROOT / "site"


def convert_notebooks():
    """将 docs/ 下的 notebook 转换为 docs_md/ 下的 markdown。"""
    # 动态导入 gen_doc，避免循环依赖
    sys.path.insert(0, str(DOCS_SRC / "tools"))
    from gen_doc import main as gen_doc_main

    print("=" * 50)
    print("步骤 1/3：转换 notebook → markdown")
    print("=" * 50)
    gen_doc_main(target_dir=str(DOCS_MD), source_dir=str(DOCS_SRC))

    # 将根目录的 通则和约定.md 复制为 index.md（MkDocs 首页）
    src_index = DOCS_MD / "通则和约定.md"
    if src_index.exists():
        shutil.copy2(src_index, DOCS_MD / "index.md")
        print(f"首页: {src_index} → {DOCS_MD / 'index.md'}")


def build_site():
    """调用 mkdocs build 构建静态站点。"""
    print()
    print("=" * 50)
    print("步骤 2/3：构建 MkDocs 站点")
    print("=" * 50)
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--clean"],
        cwd=str(ROOT),
        check=True,
    )
    print(f"\n站点已生成至: {SITE_DIR}")


def serve():
    """启动本地预览服务器。"""
    convert_notebooks()
    print()
    print("=" * 50)
    print("启动本地预览: http://127.0.0.1:8000")
    print("=" * 50)
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "serve"],
        cwd=str(ROOT),
    )


def clean():
    """清理生成的文件。"""
    for d in [DOCS_MD, SITE_DIR]:
        if d.exists():
            shutil.rmtree(d)
            print(f"已删除: {d}")


def main():
    parser = argparse.ArgumentParser(description="构建 QuantStudio 文档站点")
    parser.add_argument("--serve", action="store_true", help="启动本地预览服务器")
    parser.add_argument("--clean", action="store_true", help="清理生成的文件")
    args = parser.parse_args()

    if args.clean:
        clean()
        return

    if args.serve:
        serve()
        return

    convert_notebooks()
    build_site()
    print()
    print("=" * 50)
    print("构建完成！")
    print(f"  静态站点: {SITE_DIR}")
    print(f"  本地预览: python -m mkdocs serve")
    print("=" * 50)


if __name__ == "__main__":
    main()
