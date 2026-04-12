# -*- coding: utf-8 -*-
"""基于 notebook 生成 QuantStudio Skill"""
from pathlib import Path

from traitlets.config import Config
from nbconvert import MarkdownExporter
from nbconvert.preprocessors import TagRemovePreprocessor, ExecutePreprocessor
from nbconvert.preprocessors import Preprocessor
import nbformat

from QuantStudio import __QS_MainPath__

__READ_ME__ = str(Path(__QS_MainPath__).parent / "README.md")
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


def main(target_dir:str, source_dir:str=__NB_PATH__, readme_path:str=__READ_ME__, exec_nb:bool=False):
    # 生成 references
    c = Config()
    # 配置预处理器
    Preprocessors = []
    # 执行 notebook
    if exec_nb: Preprocessors.append(ExecutePreprocessor)
    # 移除示例数据 Cell
    c.TagRemovePreprocessor.remove_cell_tags = {'demo_data'}
    Preprocessors.append(TagRemovePreprocessor)
    # 移除空 Cell
    Preprocessors.append(RemoveEmptyCellsPreprocessor)
    # 配置导出器
    c.MarkdownExporter.preprocessors = Preprocessors
    Exporter = MarkdownExporter(config=c)
    # 导出 ipynb 为 md
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    ReferencesDir = target_dir / "references"
    for iFile in source_dir.rglob("*.ipynb"):
        if not iFile.is_file(): continue
        iTargetFile = (ReferencesDir / iFile.relative_to(source_dir)).with_suffix(".md")
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
    
    # 生成 SKILL.md
    with open(readme_path, mode="r", encoding="utf-8") as f:
        Skill = f.read()
    # 替换其中的链接路径
    Skill = Skill.replace(f"./{source_dir.parts[-1]}", "./references")
    Skill = Skill.replace(".ipynb", ".md")
    # 附加 META 信息
    Meta = """---
name: quantstudio
description: QuantStudio 投研平台使用说明，基于 QuantStudio 进行量化研发时使用该说明，内容包括因子数据访问、因子开发、因子测试、策略回测、风险模型、风险数据访问以及组合优化等
license: MIT
metadata:
  author: 麦冬
---"""
    Skill = Meta + "\n\n" + Skill
    with open(target_dir / "SKILL.md", mode="w", encoding="utf-8") as f:
        f.write(Skill)

if __name__=="__main__":
    main(target_dir=r"C:\Users\hst\Project\QSAgent\QSAgent\knowledge\skill\quantstudio")