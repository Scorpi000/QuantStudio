# -*- coding: utf-8 -*-
"""QuantStudio 使用说明（qs_help）的离线生成。

对应 QSAgent 侧 `jy_base_doc` MCP 的 `query_qs_read_data_help` /
`query_qs_get_factor_help` 两个工具。原实现从 KDB 的 `jy_base_doc` 表读
`qs_info`（由爬虫 + JYDBInfo 比对预先生成），本模块改为**离线**从
`QuantStudio/Resource/JYDBInfo.xlsx` 现场生成，不连任何数据库。

关键映射：
- 表：聚源物理表名(`base_table_name`) ↔ JYDBInfo 的 `DBTableName`
- 字段：聚源 `columnName` ↔ JYDBInfo 的 `DBFieldName`；
  因子名取 JYDBInfo 的 `FieldName`（即 QuantStudio 中的因子名）
- 参数：JYDBInfo 的 `TableClass` ↔ JYDB 模块的 `_<TableClass>` 类，
  其 `__QS_ArgClass__` 的字段名与默认值即 getTable(args=...) 的参数说明
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# 参数清单的兜底：正常从因子表类的参数模型（SQL_*Table.__QS_ArgClass__）取
# 真实参数名与默认值，仅在 QuantStudio 不可用或 TableClass 未知时退回本表。
_TABLE_TYPE_ARGS: dict[str, list[str]] = {
    "Default": ["FilterCondition", "DTField", "IDField"],
    "WideTable": [
        "FilterCondition", "DTField", "IDField", "LookBack", "OnlyLookBackDT",
        "PublDTField", "MultiMapping", "Operator", "OperatorDataType",
    ],
    "FeatureTable": ["FilterCondition", "DTField", "IDField"],
    "FinancialTable": [
        "FilterCondition", "DTField", "IDField", "ReportDate", "CalcType",
        "YearLookBack", "PeriodLookBack", "PublDTField",
    ],
    "MappingTable": [
        "FilterCondition", "DTField", "IDField", "MultiMapping",
        "EndDTField", "EndDTIncluded",
    ],
}

_READ_DATA_TEMPLATE = """在 QuantStudio 中获取表中数据的示例代码如下:
```python
import datetime as dt
import QuantStudio.api as QS

JYDB = QS.Factor.JYDB().connect()# 创建因子库对象并连接
FT = JYDB.getTable(table_name="{{ table_name }}", args={})# 获取因子表对象
Data = FT.readData(factor_names={{ factor_names }}, ids=["000001.SZ"], dts=[dt.datetime(2025, 12, 1), dt.datetime(2025, 12, 2)])# 读取表中数据
```
其中 factor_names 输入的是因子名称列表(因子名称从表的字段列表里选择), ids 输入的是证券代码列表, dts 输入的是时间列表
返回的数据 Data 是三维的 Panel 格式，比如 Data["{{ factor_name }}"] 可获取因子 {{ factor_name }} 的数据，格式为 DataFrame, index 是时间, columns 是证券代码
getTable 方法的输入变量 args 为表的参数集, 常用可选参数有:
{{ arg_info }}"""

_GET_FACTOR_TEMPLATE = """在 QuantStudio 中获取表中因子对象的示例代码如下:
```python
import QuantStudio.api as QS

JYDB = QS.Factor.JYDB().connect()# 创建因子库对象并连接
FT = JYDB.getTable(table_name="{{ table_name }}", args={})# 获取因子表对象
F = FT.getFactor(factor_name="{{ factor_name }}")# 获取因子对象
```
其中 F 是创建的因子对象(因子名称从表的字段列表里选择), getTable 方法的输入变量 args 为表的参数集, 可选参数如下(每行依次为参数名、中文名、默认值):
{{ arg_info }}"""

# readData 的 dts 需要具体日期；模板里写死一个示例日期会误导，
# 这里保持与原实现一致（原实现也是渲染固定示例）。
_NOT_FOUND = "QuantStudio 不支持使用该表 {table}"


@lru_cache(maxsize=1)
def _load_jydb_info():
    """加载 JYDBInfo.xlsx，返回 (TableInfo, FactorInfo) 两个 DataFrame。

    结果缓存：xlsx 有 495 表 / 12677 字段，解析一次约百毫秒级，值得复用。
    失败时返回 None，由调用方给出可读提示。
    """
    try:
        import logging as _logging
        import os

        from QuantStudio import __QS_MainPath__
        from QuantStudio.Factor.JYDB import _importInfo

        xlsx = os.path.join(__QS_MainPath__, "Resource", "JYDBInfo.xlsx")
        if not os.path.exists(xlsx):
            logger.warning("JYDBInfo.xlsx 不存在: %s", xlsx)
            return None
        # _importInfo 会刷大量 INFO 日志，这里压到 WARNING 以免污染 MCP stdio 输出
        quiet = _logging.getLogger("jy_doc.jydbinfo")
        quiet.setLevel(_logging.WARNING)
        table_info, factor_info, _, _ = _importInfo(None, xlsx, quiet)
        return table_info, factor_info
    except Exception as e:
        logger.warning("加载 JYDBInfo.xlsx 失败: %s: %s", type(e).__name__, e)
        return None


def _lookup(base_table_name: str):
    """按聚源物理表名查 JYDBInfo。

    Returns:
        (table_name, table_row, factor_frame) 或 None。
        一张物理表可能对应多个 QuantStudio 表名（同表不同 args/后缀注册），
        此处取第一行，并在调用方提示存在多个。
    """
    if not base_table_name:
        return None
    info = _load_jydb_info()
    if info is None:
        return None
    table_info, factor_info = info

    hits = table_info[table_info["DBTableName"] == base_table_name]
    if hits.empty:
        return None

    table_name = hits.index[0]
    if table_name not in factor_info.index.get_level_values(0):
        factor_frame = factor_info.iloc[0:0]  # 空表，字段结构完整
    else:
        # FI.loc[表名] 返回以 FieldName 为索引的 DataFrame
        factor_frame = factor_info.loc[[table_name]].reset_index()
    return table_name, hits.iloc[0], factor_frame


# QuantStudio 内部生成的辅助因子名，不应出现在 getFactor/readData 示例中
_AUX_FACTOR_NAMES = {"JYID", "JSID", "ID", "更新时间", "发布时间"}

# 不产出因子的 FieldType（ID/时点是坐标轴，AnnDate/EndDate 等是取数参数）
_NON_FACTOR_FIELD_TYPES = {
    "ID", "Date", "AnnDate", "EndDate", "ReportDate",
    "AdjustType", "Condition", "Factor", "Period", "Institute", "Analyst", "CurSign",
}


def _is_factor_row(row) -> bool:
    """判断 FactorInfo 的一行是否是一个可供 getFactor 的因子。

    过滤掉 FieldType 为空（不参与因子化）以及 ID/时点/参数类字段。
    """
    ftype = row.get("FieldType")
    if not isinstance(ftype, str) or not ftype.strip():
        return False
    if ftype in _NON_FACTOR_FIELD_TYPES:
        return False
    name = row.get("FieldName")
    return isinstance(name, str) and bool(name) and name not in _AUX_FACTOR_NAMES


def _factor_names(factor_frame) -> list[str]:
    """从 FactorInfo 中提取因子名列表（只含真正的因子字段）。"""
    if factor_frame is None or factor_frame.empty:
        return []
    names = [row.get("FieldName") for _, row in factor_frame.iterrows() if _is_factor_row(row)]
    return sorted(set(names))


@lru_cache(maxsize=None)
def _table_arg_fields(table_class: str) -> Optional[dict]:
    """取某 TableClass 对应因子表类的参数模型字段 {参数名: FieldInfo}。

    参数模型即 FactorUtils 中 `SQL_*Table.__QS_ArgClass__`，其字段名就是
    getTable(args=...) 可用的参数名。表类可按 TableClass 直接拼出模块级
    ``_<TableClass>``（如 ``_WideTable``），故无需硬编码类型表。

    只保留 ``repr=True`` 的字段：框架自身用该标志区分面向使用者的参数与内部
    参数（如 TaskExecutor、各索引设置），故这里照此口径展示。

    Returns:
        参数字段字典；QuantStudio 不可用或类型未知时返回 None。
    """
    if not table_class:
        return None
    try:
        import QuantStudio.Factor.JYDB as JYDBMod

        cls = getattr(JYDBMod, "_" + table_class, None)
        if cls is None:
            return None
        fields = cls.__QS_ArgClass__.model_fields
        return {name: field for name, field in fields.items() if field.repr}
    except Exception as e:
        logger.warning("获取表类型 %s 的参数模型失败: %s: %s", table_class, type(e).__name__, e)
        return None


def _field_arg_title(field, arg_name: str) -> str:
    """解析单个参数的展示名（优先用参数模型里的中文名）。"""
    title = getattr(field, "title", None)
    if isinstance(title, str) and title.strip():
        return title.strip()
    return arg_name


def _format_arg_default(field) -> str:
    """把参数模型的字段默认值渲染成简短文本。

    无默认值（必填）时返回空串，由调用方另行标注。
    """
    if field.is_required():
        return ""
    default = field.get_default(call_default_factory=True)
    if isinstance(default, str):
        return f"'{default}'" if default else "''"
    if default is None:
        return "None"
    return repr(default)


def _arg_info(table_row) -> str:
    """生成 args 各个参数的说明文本（含默认值）。

    参数清单以对应因子表类的参数模型为准（即 getTable(args=...) 真正接受的
    参数），再附上该表在 JYDBInfo 中配置的 DefaultArgs——后者是实际生效的
    取值，可能覆盖参数模型的默认值。
    """
    table_class = table_row.get("TableClass")
    table_class = table_class if isinstance(table_class, str) else ""
    default_args = table_row.get("DefaultArgs")
    default_args = default_args if isinstance(default_args, str) else ""
    # JYDBInfo 中的 DefaultArgs 是 dict 字面量字符串，直接 eval 有风险，
    # 仅抽取其中的参数名（形如 'ArgName': ...），只为下文“默认值”一节提供名单。
    default_names = re.findall(r"'([^']+)'\s*:", default_args)

    fields = _table_arg_fields(table_class)
    lines = []
    if fields is None:
        # 参数模型取不到时退化为静态参数名清单，仍附上表自身的默认参数
        args = _TABLE_TYPE_ARGS.get(table_class, _TABLE_TYPE_ARGS["Default"])
        lines.append("  " + ", ".join(args))
    else:
        lines.append("  " + ", ".join(fields))
        for arg_name, field in fields.items():
            description = getattr(field, "description", None)
            line = f"    - {arg_name}({_field_arg_title(field, arg_name)}): "
            if field.is_required():
                line += "必填参数"
            else:
                line += f"默认 {_format_arg_default(field)}"
            if isinstance(description, str) and description.strip():
                desc = description.strip().replace("\n", " ")
                line += f"；{desc[:200]}" + ("..." if len(desc) > 200 else "")
            lines.append(line)
    if default_args:
        # DefaultArgs 给的是取值而非参数名，且可能覆盖参数模型的默认值
        # （如参数模型 AdjustType 默认空串，此处为 '2,1'），故原文给出，
        # 仅另行列出来源字典中出现过的参数名。
        lines.append("调用 getTable 时该表在 JYDBInfo 中配置的默认参数（覆盖上表默认值）")
        if default_names:
            lines.append("（涉及参数: " + ", ".join(dict.fromkeys(default_names)) + "）")
        lines.append(f"  {default_args}")
    return "\n".join(lines)


def read_data_help(base_table_name: str) -> str:
    """生成某表在 QuantStudio 中读取数据的说明。

    Args:
        base_table_name: 聚源物理表名（jy_doc 索引里的 base_table_name）

    Returns:
        示例代码 + args 参数说明；不支持时返回提示文本
    """
    found = _lookup(base_table_name)
    if found is None:
        return _NOT_FOUND.format(table=base_table_name)

    table_name, table_row, factor_frame = found
    names = _factor_names(factor_frame)
    if not names:
        return (
            f"表 '{base_table_name}' 已在 JYDBInfo 中注册为 '{table_name}'，"
            f"但未配置任何因子字段（FactorInfo 中 FieldType 均为空），无法读取数据。"
        )

    return _READ_DATA_TEMPLATE.replace("{{ table_name }}", table_name).replace(
        "{{ factor_names }}", str(names[:2])
    ).replace("{{ factor_name }}", names[0]).replace("{{ arg_info }}", _arg_info(table_row))


def get_factor_help(base_table_name: str) -> str:
    """生成某表在 QuantStudio 中获取因子对象的说明。

    Args:
        base_table_name: 聚源物理表名（jy_doc 索引里的 base_table_name）

    Returns:
        示例代码 + args 参数说明；不支持时返回提示文本
    """
    found = _lookup(base_table_name)
    if found is None:
        return _NOT_FOUND.format(table=base_table_name)

    table_name, table_row, factor_frame = found
    names = _factor_names(factor_frame)
    if not names:
        return (
            f"表 '{base_table_name}' 已在 JYDBInfo 中注册为 '{table_name}'，"
            f"但未配置任何因子字段（FactorInfo 中 FieldType 均为空），无法获取因子对象。"
        )

    return _GET_FACTOR_TEMPLATE.replace("{{ table_name }}", table_name).replace(
        "{{ factor_name }}", names[0]
    ).replace("{{ arg_info }}", _arg_info(table_row))


def factor_name_map(base_table_name: str) -> dict[str, str]:
    """返回 {聚源物理字段名: QuantStudio 因子名} 映射。

    用于把聚源字段名适配成 JYDB 因子名。仅含真正的因子字段
    （FieldType 为因子/Value/Group 等取值字段），不含 ID/时点等坐标字段。
    """
    found = _lookup(base_table_name)
    if found is None:
        return {}
    _, _, factor_frame = found
    if factor_frame is None or factor_frame.empty:
        return {}

    mapping = {}
    for _, row in factor_frame.iterrows():
        if not _is_factor_row(row):
            continue
        db_field, field_name = row.get("DBFieldName"), row.get("FieldName")
        if isinstance(db_field, str) and db_field and isinstance(field_name, str) and field_name:
            mapping[db_field] = field_name
    return mapping


def preload() -> bool:
    """在主线程预热 JYDBInfo 的加载（解析 xlsx + 拉起原生扩展）。

    必须在 MCP 服务启动阶段、事件循环之前调用：`_importInfo` 会导入
    pandas/numpy 等原生扩展，若首次导入发生在 FastMCP 的工具线程中，会触发
    Windows 的 DLL 加载器锁死锁（详见 server._preimport_native_deps）。
    同时把解析结果填入 lru_cache，避免首次调用时重复解析；并预热所有表类型的
    参数模型，使参数说明也一并生成在本次调用中。

    Returns:
        加载成功返回 True，失败（如 QuantStudio 不可用）返回 False。
    """
    info = _load_jydb_info()
    if info is None:
        return False
    table_info = info[0]
    for table_class in table_info["TableClass"].dropna().unique():
        _table_arg_fields(str(table_class))
    return True
