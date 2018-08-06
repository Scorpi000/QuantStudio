# coding=utf-8
"""基于 xlwings 模块的 Excel 功能函数"""

import numpy as np
import pandas as pd


# 复制图表
def copyChart(xl_book,sheet_name,chart_name,target_row,target_col,target_sheet_name=None,new_chart_name=None):
    CurSheet = xl_book.sheets[sheet_name]
    Chrt = CurSheet.charts[chart_name].api[1].ChartArea.Copy()
    if target_sheet_name is None:
        target_sheet_name = sheet_name
    TargetSheet = xl_book.sheets(target_sheet_name)
    TargetSheet.api.Paste(TargetSheet[target_row,target_col].api)
    if new_chart_name is not None:
        TargetSheet.charts[TargetSheet.charts.count-1].name = new_chart_name
    return TargetSheet.charts[TargetSheet.charts.count-1]