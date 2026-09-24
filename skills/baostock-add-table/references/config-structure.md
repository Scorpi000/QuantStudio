# BaoStockDBInfo.xlsx 配置文件结构

配置文件路径：`QuantStudio/Resource/BaoStockDBInfo.xlsx`

## Sheet 1: TableInfo

| 列名 | 说明 | 示例 |
|------|------|------|
| TableName | 表的中文名称（主键） | 季度盈利能力 |
| DBTableName | BaoStock API 函数名 | query_profit_data |
| TableClass | 表类型类名 | QuarterTable |
| DefaultArgs | 默认参数 JSON | {"IDAdj":"前缀","DTFmt":"%Y-%m-%d"} |
| SecurityType | 证券类型 | A股 |
| Description | 表的简要描述 | 季频盈利能力指标 |
| URL | API 文档链接 | http://baostock.com/... |

## Sheet 2: FactorInfo

复合主键：(TableName, FieldName)

| 列名 | 说明 | 示例 |
|------|------|------|
| TableName | 所属表名 | 季度盈利能力 |
| FieldName | 字段名 | roeAvg |
| DataType | 数据类型 | float |
| FieldType | 字段类型 | 因子 |
| Supplementary | 补充信息（通常为空） | |
| Description | 字段描述 | 净资产收益率(平均)(%) |

### FieldType 取值

- `ID`: 证券代码字段（如 code）
- `Date`: 日期字段，用于时间对齐
- `因子`: 数据字段

### DataType 取值

- `string`: 字符串
- `datetime`: 日期时间
- `float`: 浮点数

## Sheet 3: ArgInfo

复合主键：(TableName, ArgName)

| 列名 | 说明 | 示例 |
|------|------|------|
| TableName | 所属表名 | 季度盈利能力 |
| ArgName | 参数名 | year |
| DataType | 参数数据类型 | int |
| FieldType | 参数类型 | Year |
| DefaultValue | 默认值 | 2024 |
| ArgInfo | 参数约束 JSON | {"arg_type":"SingleOption","option_range":[1,2,3,4]} |
| Description | 参数描述 | 统计年份 |

### FieldType 取值

- `ID`: 证券代码参数
- `Date`: 日期参数
- `StartDate`/`EndDate`: 起止日期参数
- `Year`: 年份参数
- `Quarter`: 季度参数
- `Field`: 字段列表参数
- `QSArg`: 可配置的 API 参数

### ArgInfo JSON 格式（仅 QSArg 类型）

```json
{
  "arg_type": "SingleOption",
  "option_range": ["d", "w", "m", "5", "15", "30", "60"]
}
```

- `arg_type`: 参数类型，目前仅支持 `SingleOption`
- `option_range`: 可选值列表

## 使用 Python 更新配置

```python
import pandas as pd

xlsx_path = 'QuantStudio/Resource/BaoStockDBInfo.xlsx'

# 读取
TableInfo = pd.read_excel(xlsx_path, sheet_name='TableInfo').set_index(['TableName'])
FactorInfo = pd.read_excel(xlsx_path, sheet_name='FactorInfo').set_index(['TableName', 'FieldName'])
ArgInfo = pd.read_excel(xlsx_path, sheet_name='ArgInfo').set_index(['TableName', 'ArgName'])

# 添加新表配置...

# 写回（必须同时写入所有 sheets）
with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    TableInfo.reset_index().to_excel(writer, sheet_name='TableInfo', index=False)
    FactorInfo.reset_index().to_excel(writer, sheet_name='FactorInfo', index=False)
    ArgInfo.reset_index().to_excel(writer, sheet_name='ArgInfo', index=False)
```
