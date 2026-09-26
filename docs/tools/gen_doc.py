# -*- coding: utf-8 -*-
"""基于 notebook 生成 markdown 文档，供 MkDocs 使用。

功能：
* 移除空单元格
* 从首个 markdown 标题提取 frontmatter title
* 将 .ipynb 内部链接转换为 .md 链接
* 生成 Skill 参考文档时，按 cell tag 过滤代码单元格、剥离输出中的图片
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

# 生成 Skill 参考文档时，保留代码单元格的标记
SKILL_TAG = "skill"

# 图片剥离后留下的占位文本
IMAGE_PLACEHOLDER = "[图片已省略]"

# 匹配 HTML 输出中内联的图片标签（图片以 base64 编码内嵌，体积可达数百 KB）
_IMG_PATTERN = re.compile(r"<img\b[^>]*>", re.IGNORECASE)

# 匹配 QuantStudio 日志行（含时间戳），这类输出与运行环境绑定，对 Skill 无价值
_LOG_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ \| QS \|.*$\n?", re.MULTILINE)

# 匹配本机绝对路径（Windows 用户目录），避免把本机环境写入 Skill
_PATH_PATTERN = re.compile(r"[A-Za-z]:\\+Users\\+[^\\\s\"']+")


class RemoveEmptyCellsPreprocessor(Preprocessor):
    """移除空单元格的预处理器"""

    def preprocess(self, nb, resources):
        nb.cells = [c for c in nb.cells if c['source'].strip()]
        return nb, resources


class SkillCellFilterPreprocessor(Preprocessor):
    """仅保留标记了 skill tag 的代码单元格，供生成 Skill 参考文档使用。

    未标记的代码单元格通常是与文档叙事无关的胶水代码（导入、初始化、环境设置），
    对 Skill 无价值。markdown 单元格一律保留。
    """

    def preprocess(self, nb, resources):
        nb.cells = [
            c for c in nb.cells
            if c['cell_type'] != 'code' or SKILL_TAG in c.get('metadata', {}).get('tags', [])
        ]
        return nb, resources


class ImageStripPreprocessor(Preprocessor):
    """剥离输出中的图片，仅保留文本内容。

    图片有两处来源：
    * 独立的多媒体输出，存于 ``output.data`` 的 ``image/*`` 键
    * HTML 输出中内联的 ``<img>`` 标签（``text/html`` 键的字符串内，base64 内嵌）

    后者是主要来源——报告类输出常把整张图 base64 内嵌进 HTML。
    """

    def preprocess(self, nb, resources):
        for iCell in nb.cells:
            for iOutput in iCell.get('outputs', []):
                iData = iOutput.get('data', {})
                for iKey in [k for k in iData if k.startswith('image/')]:
                    iData.pop(iKey)
                if 'text/html' in iData:
                    iData['text/html'] = _IMG_PATTERN.sub(
                        IMAGE_PLACEHOLDER, iData['text/html']
                    )
        return nb, resources


class ScrubEnvironmentNoisePreprocessor(Preprocessor):
    """清除输出中与运行环境绑定的内容：QuantStudio 日志行、本机绝对路径。

    这类内容依赖执行时的机器与时刻，写入 Skill 后不仅无价值，还会暴露本机路径。
    """

    def preprocess(self, nb, resources):
        for iCell in nb.cells:
            for iOutput in iCell.get('outputs', []):
                if 'text' in iOutput:
                    iOutput['text'] = self._scrub(iOutput['text'])
                iData = iOutput.get('data', {})
                for iKey in ('text/plain', 'text/html'):
                    if iKey in iData:
                        iData[iKey] = self._scrub(iData[iKey])
        return nb, resources

    @staticmethod
    def _scrub(text):
        if isinstance(text, list):
            text = ''.join(text)
        text = _LOG_PATTERN.sub('', text)
        return _PATH_PATTERN.sub('[本机路径]', text)


class DropExecutionCountPreprocessor(Preprocessor):
    """清空执行计数，去除本机执行痕迹。"""

    def preprocess(self, nb, resources):
        for iCell in nb.cells:
            iCell['execution_count'] = None
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
    skill_mode: bool = False,
    strip_images: bool = False,
    drop_exec_count: bool = False,
):
    """将 source_dir 下的 notebook 转换到 target_dir。

    Args:
        target_dir: 输出目录
        source_dir: notebook 源目录
        doc_fmt: 输出格式，Markdown 或 reStructuredText
        postprocess: 是否进行后处理（添加 frontmatter、转换链接），仅 Markdown 格式有效
        skill_mode: 是否生成 Skill 参考文档。启用后仅保留标记了 skill tag 的代码单元格，
            并清除输出中的日志行与本机路径
        strip_images: 是否剥离输出中的图片（含 HTML 输出内联的图片标签）
        drop_exec_count: 是否清空执行计数
    """
    c = Config()

    Preprocessors = [RemoveEmptyCellsPreprocessor]
    if skill_mode:
        Preprocessors.extend([
            SkillCellFilterPreprocessor,
            ScrubEnvironmentNoisePreprocessor,
        ])
    if strip_images:
        Preprocessors.append(ImageStripPreprocessor)
    if drop_exec_count:
        Preprocessors.append(DropExecutionCountPreprocessor)

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