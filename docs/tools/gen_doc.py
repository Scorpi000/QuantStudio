# -*- coding: utf-8 -*-
"""基于 notebook 生成 markdown 文档，供 MkDocs 使用。

功能：
* 移除空单元格
* 从首个 markdown 标题提取 frontmatter title
* 将 .ipynb 内部链接转换为 .md 链接
"""
import re
from pathlib import Path
from typing import Literal

from traitlets.config import Config
from nbconvert import MarkdownExporter, RSTExporter
from nbconvert.preprocessors import Preprocessor
import nbformat

from QuantStudio import __QS_MainPath__


__NB_PATH__ = str(Path(__QS_MainPath__).parent / "docs")


class RemoveEmptyCellsPreprocessor(Preprocessor):
    """移除空单元格的预处理器"""

    def preprocess(self, nb, resources):
        nb.cells = [c for c in nb.cells if c['source'].strip()]
        return nb, resources


def _postprocess_markdown(content: str) -> str:
    """对转换后的 markdown 进行后处理：添加 frontmatter、转换链接。"""
    # 1. 从首个 # 标题提取 title
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else ""

    # 2. 将 .ipynb 链接转为 .md 链接
    content = re.sub(r'\.ipynb([)#])', r'.md\1', content)
    # 处理 .ipynb#anchor 形式
    content = re.sub(r'\.ipynb(#)', r'.md\1', content)

    # 3. 添加 frontmatter
    frontmatter = f"---\ntitle: \"{title}\"\n---\n\n"
    return frontmatter + content


def main(
    target_dir: str,
    source_dir: str = __NB_PATH__,
    doc_fmt: Literal["markdown", "Markdown", "reStructuredText", "rst"] = "Markdown",
    postprocess: bool = True,
):
    """将 source_dir 下的 notebook 转换到 target_dir。

    Args:
        target_dir: 输出目录
        source_dir: notebook 源目录
        doc_fmt: 输出格式，Markdown 或 reStructuredText
        postprocess: 是否进行后处理（添加 frontmatter、转换链接），仅 Markdown 格式有效
    """
    c = Config()

    Preprocessors = [RemoveEmptyCellsPreprocessor]

    if doc_fmt.lower() == "markdown":
        c.MarkdownExporter.preprocessors = Preprocessors
        Exporter = MarkdownExporter(config=c)
        Suffix = ".md"
    elif doc_fmt.lower() in ("restructuredtext", "rst"):
        c.RSTExporter.preprocessors = Preprocessors
        Exporter = RSTExporter(config=c)
        Suffix = ".rst"
    else:
        raise Exception(f"不支持的参数 doc_fmt 取值: {doc_fmt}")

    source_dir, target_dir = Path(source_dir), Path(target_dir)
    for iFile in source_dir.rglob("*.ipynb"):
        if not iFile.is_file():
            continue
        if ".ipynb_checkpoints" in iFile.parts:
            continue
        iTargetFile = (target_dir / iFile.relative_to(source_dir)).with_suffix(Suffix)
        iTargetFile.parent.mkdir(parents=True, exist_ok=True)
        # 读取 notebook
        with open(iFile, mode="r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        # 转换
        DocContent, _ = Exporter.from_notebook_node(nb, {})
        # 后处理
        if postprocess and doc_fmt.lower() == "markdown":
            DocContent = _postprocess_markdown(DocContent)
        # 保存
        with open(iTargetFile, mode="w", encoding="utf-8") as f:
            f.write(DocContent)
        print(f"{iFile} -> {iTargetFile} 转换完毕")


if __name__ == "__main__":
    main(target_dir=r"C:\Users\hst\Project\Script\QSDoc1", doc_fmt="Markdown")