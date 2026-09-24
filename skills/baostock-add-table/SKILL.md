---
name: baostock-add-table
description: |
  添加 BaoStock 表到 QuantStudio 的 BaoStockDB 因子库。当用户提供 BaoStock API 文档 HTML 文件并要求添加新表时使用此 skill。
  触发关键词：添加 BaoStock 表、baostock 新表、BaoStockDB 配置、query_xxx_data
---

# BaoStock 添加表 Skill

帮助用户将新的 BaoStock 数据表添加到 QuantStudio 的 BaoStockDB 因子库中。

## 前提条件

用户必须提供 BaoStock API 文档的 HTML 文件（从 baostock.com 下载的页面）。

## 工作流程

### 1. 读取并解析文档

使用 `grep` 或 `Read` 工具从用户提供的 HTML 文件中提取：

- **API 函数名**：如 `query_profit_data`、`query_dividend_data`
- **参数列表**：参数名、类型、是否必填、默认值
- **返回字段**：字段名、数据类型、中文描述、算法说明

### 2. 确定表类型

根据 API 参数特征选择合适的表类型：

| 表类型 | 参数特征 | 示例 |
|--------|---------|------|
| `DTTable` | 按单个日期查询，参数含 `date` | `query_stock_industry` |
| `DTRangeTable` | 按日期区间查询，参数含 `start_date` + `end_date` | `query_history_k_data_plus` |
| `QuarterTable` | 按年份+季度查询，参数含 `year` + `quarter` | `query_profit_data` |
| `YearTable` | 仅按年份查询，参数含 `year`（无 quarter） | `query_dividend_data` |

### 3. 更新配置文件

编辑 `QuantStudio/Resource/BaoStockDBInfo.xlsx`：

**TableInfo 新增行：**
- `TableName`: 表的中文名称
- `DBTableName`: BaoStock API 函数名
- `TableClass`: 表类型（从步骤2确定）
- `DefaultArgs`: 默认参数 JSON，如 `{"IDAdj":"前缀","DTFmt":"%Y-%m-%d"}`
- `SecurityType`: 证券类型，通常为 "A股"
- `Description`: 表的简要描述
- `URL`: API 文档链接

**FactorInfo 新增行（每字段一行）：**
- `TableName`: 所属表名
- `FieldName`: 字段名
- `DataType`: 数据类型（string/datetime/float）
- `FieldType`: 字段类型
  - `ID`: 证券代码字段（如 code）
  - `Date`: 日期字段（用于时间对齐，通常选择统计截止日期或除权除息日）
  - `因子`: 其他数据字段
- `Description`: 字段描述，包含算法说明（如有）

**ArgInfo 新增行（每参数一行）：**
- `TableName`: 所属表名
- `ArgName`: 参数名
- `DataType`: 参数数据类型（str/int）
- `FieldType`: 参数类型
  - `ID`: 证券代码参数
  - `Year`: 年份参数
  - `Quarter`: 季度参数
  - `Date`: 日期参数
  - `StartDate`/`EndDate`: 起止日期参数
  - `Field`: 字段列表参数
  - `QSArg`: 可配置的 API 参数（如 frequency、adjustflag）
- `DefaultValue`: 默认值
- `ArgInfo`: 参数约束信息 JSON（仅 QSArg 类型需要）
- `Description`: 参数描述

### 4. 检查是否需要新表类型

如果现有表类型不满足需求，需要在 `QuantStudio/Factor/BaoStockDB.py` 中添加新的表类型类。

## 使用示例

```
用户：帮我添加 BaoStock 的季频营运能力表，文档在这里：D:\HST\baostock_operate.html

Claude：
1. 读取 HTML 文档，解析出 query_operation_data 的参数和返回字段
2. 确定表类型为 QuarterTable（因为有 year + quarter 参数）
3. 更新 BaoStockDBInfo.xlsx 添加配置
4. 验证功能正常
```

## 验证

添加完成后，使用以下代码验证：

```python
import datetime as dt
from QuantStudio.Factor.BaoStockDB import BaoStockDB

BSDB = BaoStockDB().connect()
print("支持的表:", BSDB.TableNames)

FT = BSDB.getTable("新表名称")
print("表类型:", type(FT).__name__)
print("因子列表:", FT.FactorNames)

Data = FT.readData(
    factor_names=["factor1", "factor2"],
    ids=["600000.SH"],
    dts=[dt.datetime(2024, 12, 31)]
)
print(Data)
```

## 注意事项

- 字段描述必须来自官方文档，不要猜测
- 算法说明也要记录到 Description 字段
- Date 类型字段的选择：优先选择统计截止日期（如 statDate）或除权除息日期（如 dividOperateDate）
