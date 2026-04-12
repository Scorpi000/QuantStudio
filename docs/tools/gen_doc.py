# -*- coding: utf-8 -*-
"""基于 notebook 生成文档"""
from pathlib import Path
from typing import Literal

from traitlets.config import Config
from nbconvert import MarkdownExporter, RSTExporter
from nbconvert.preprocessors import Preprocessor
import nbformat

from QuantStudio import __QS_MainPath__


__NB_PATH__ = str(Path(__QS_MainPath__).parent / "docs")

class RemoveEmptyCellsPreprocessor(Preprocessor):
    """一个用于移除空单元格的预处理器"""

    def preprocess(self, nb, resources):
        Cells = []
        for iCell in nb.cells:
            if iCell['source'].strip():
                Cells.append(iCell)
        nb.cells = Cells
        return nb, resources


def main(target_dir:str, source_dir:str=__NB_PATH__, doc_fmt:Literal["Markdown", "reStructuredText"]="markdown"):
    c = Config()

    # 配置预处理器
    Preprocessors = []
    Preprocessors.append(RemoveEmptyCellsPreprocessor)

    # 配置导出器
    if doc_fmt=="Markdown":
        c.MarkdownExporter.preprocessors = Preprocessors
        Exporter = MarkdownExporter(config=c)
        Suffix = ".md"
    elif doc_fmt=="reStructuredText":
        c.RSTExporter.preprocessors = Preprocessors
        Exporter = RSTExporter(config=c)
        Suffix = ".rst"
    else:
        raise Exception(f"不支持的参数 doc_fmt 取值: {doc_fmt}")
    
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    for iFile in source_dir.rglob("*.ipynb"):
        if not iFile.is_file(): continue
        iTargetFile = (target_dir / iFile.relative_to(source_dir)).with_suffix(Suffix)
        iTargetFile.parent.mkdir(parents=True, exist_ok=True)
        # 读取 notebook
        with open(iFile, mode="r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        # 转换
        DocContent, _ = Exporter.from_notebook_node(nb, {})
        # 保存
        with open(iTargetFile, mode="w", encoding="utf-8") as f:
            f.write(DocContent)
        print(f"{iFile} -> {iTargetFile} 转换完毕")

if __name__=="__main__":
    main(target_dir=r"C:\Users\hst\Project\Script\QSDoc1", doc_fmt="Markdown")