# -*- coding: utf-8 -*-
"""验证 Notebook 在普通文档和 Skill 导出中的独立排除规则。"""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gen_doc", ROOT / "docs/tools/gen_doc.py")
GEN_DOC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN_DOC)


class TestNotebookExport(unittest.TestCase):
    """通过实际 Markdown 导出验证内容与输出是否被正确收录。"""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.path = self.source / "示例.ipynb"
        cells = [nbformat.v4.new_markdown_cell("# 导出示例")]
        for label, tags in (
            ("DEFAULT", []),
            ("MD_ONLY_EXCLUDED", ["md-excluded"]),
            ("SKILL_ONLY_EXCLUDED", ["skill-excluded"]),
            ("BOTH_EXCLUDED", ["md-excluded", "skill-excluded"]),
            ("OTHER_TAG", ["demo_data"]),
        ):
            cells.extend([
                nbformat.v4.new_markdown_cell(f"{label}_MARKDOWN", metadata={"tags": tags}),
                nbformat.v4.new_code_cell(
                    f"print('{label}_CODE')",
                    metadata={"tags": tags},
                    outputs=[nbformat.v4.new_output("stream", name="stdout", text=f"{label}_OUTPUT\n")],
                ),
                nbformat.v4.new_raw_cell(
                    f"{label}_RAW", metadata={"tags": tags, "raw_mimetype": "text/markdown"},
                ),
            ])
        nbformat.write(nbformat.v4.new_notebook(cells=cells), self.path)

    def export(self, skill_mode):
        """导出测试 Notebook，返回生成的 Markdown 内容。

        Args:
            skill_mode: 是否使用 Skill 导出规则。

        Returns:
            生成的 Markdown 文本。
        """
        target = self.root / ("skill" if skill_mode else "markdown")
        with contextlib.redirect_stdout(io.StringIO()):
            GEN_DOC.main(str(target), source_dir=str(self.source), skill_mode=skill_mode)
        return (target / "示例.md").read_text(encoding="utf-8")

    def assert_content(self, content, included, excluded):
        """检查不同类型单元及代码输出的收录结果。"""
        for label in included:
            for suffix in ("MARKDOWN", "CODE", "OUTPUT", "RAW"):
                self.assertIn(f"{label}_{suffix}", content)
        for label in excluded:
            self.assertNotIn(label, content)

    def test_markdown_only_applies_md_excluded(self):
        self.assert_content(
            self.export(skill_mode=False),
            included=("DEFAULT", "SKILL_ONLY_EXCLUDED", "OTHER_TAG"),
            excluded=("MD_ONLY_EXCLUDED", "BOTH_EXCLUDED"),
        )

    def test_skill_only_applies_skill_excluded(self):
        self.assert_content(
            self.export(skill_mode=True),
            included=("DEFAULT", "MD_ONLY_EXCLUDED", "OTHER_TAG"),
            excluded=("SKILL_ONLY_EXCLUDED", "BOTH_EXCLUDED"),
        )

    def test_excluded_heading_does_not_set_document_title(self):
        notebook = nbformat.read(self.path, as_version=4)
        notebook.cells.insert(0, nbformat.v4.new_markdown_cell(
            "# EXCLUDED_TITLE", metadata={"tags": ["md-excluded", "skill-excluded"]},
        ))
        nbformat.write(notebook, self.path)
        for skill_mode in (False, True):
            with self.subTest(skill_mode=skill_mode):
                content = self.export(skill_mode)
                self.assertIn('title: "导出示例"', content)
                self.assertNotIn("EXCLUDED_TITLE", content)

    def test_export_does_not_change_source_notebook(self):
        original = self.path.read_bytes()
        self.export(skill_mode=False)
        self.export(skill_mode=True)
        self.assertEqual(self.path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
