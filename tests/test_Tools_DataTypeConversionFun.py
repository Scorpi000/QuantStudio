"""列表展开时保留索引、成分数据对齐及空列表标记。"""
import unittest

import pandas as pd

from QuantStudio.Tools.DataTypeConversionFun import expandListElementDataFrame


class TestExpandListElementDataFrame(unittest.TestCase):
    def test_string_index_and_aligned_component_data(self):
        """StringDtype 索引可以暂存重复标签列表，成分与权重仍逐行对应。"""
        data = pd.DataFrame({"member": [["A", "B"], ["C"]], "weight": [[1., 2.], [3.]]},
                            index=pd.Index(["fund_a", "fund_b"], dtype="string", name="fund"))
        original = data.copy(deep=True)
        result = expandListElementDataFrame(data)
        self.assertEqual(result.columns.tolist(), ["fund", "member", "weight"])
        self.assertEqual(result["fund"].tolist(), ["fund_a", "fund_a", "fund_b"])
        self.assertEqual(result["member"].tolist(), ["A", "B", "C"])
        self.assertEqual(result["weight"].tolist(), [1., 2., 3.])
        pd.testing.assert_frame_equal(data, original)

    def test_multiple_index_levels_with_different_types(self):
        """每一级索引按列表长度重复，保留字符串、数值和日期标签。"""
        for second_level in ([2025, 2026], pd.to_datetime(["2025-01-01", "2026-01-01"])):
            with self.subTest(second_level=second_level):
                index = pd.MultiIndex.from_arrays([pd.Index(["first", "second"], dtype="string"), second_level],
                                                  names=["name", "period"])
                result = expandListElementDataFrame(pd.DataFrame({"member": [["A", "B"], ["C"]]}, index=index))
                self.assertEqual(result["name"].tolist(), ["first", "first", "second"])
                self.assertEqual(result["period"].tolist(), [second_level[0], second_level[0], second_level[1]])
                self.assertEqual(result["member"].tolist(), ["A", "B", "C"])

    def test_empty_and_missing_lists_keep_existing_mask_semantics(self):
        """展开结果先列非空成分，空列表与缺失值按原约定保留或移除。"""
        data = pd.DataFrame({"member": [["A", "B"], [], None, ["C"]]},
                            index=pd.Index(["full", "empty", "missing", "single"], dtype="string", name="source"))
        for dropna in (False, True):
            with self.subTest(dropna=dropna):
                result, mask = expandListElementDataFrame(data, dropna=dropna, empty_list_mask=True)
                expected_sources = ["full", "full", "single", "empty"] + ([] if dropna else ["missing"])
                self.assertEqual(result["source"].tolist(), expected_sources)
                self.assertEqual(result["member"].iloc[:3].tolist(), ["A", "B", "C"])
                self.assertTrue(result["member"].iloc[3:].isna().all())
                self.assertEqual(mask.tolist(), [False, False, False, True] + ([] if dropna else [False]))


if __name__ == "__main__":
    unittest.main()
