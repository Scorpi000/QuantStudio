# BaoStock 表类型参考

本文档记录现有的表类型实现模式，用于在需要添加新表类型时参考。

## 现有表类型

### 1. DTTable - 单日期查询

**适用场景**：API 按单个日期查询数据

**典型 API**：`query_stock_industry(code, date)`

**配置示例**：
```
TableInfo:
  - DefaultArgs: {"IDAdj":"前缀","DTFmt":"%Y-%m-%d"}

ArgInfo:
  - code: FieldType=ID
  - date: FieldType=Date
```

**代码实现**：`QuantStudio/Factor/BaoStockDB.py` 第 164-206 行

### 2. DTRangeTable - 日期区间查询

**适用场景**：API 按日期区间查询数据，需要遍历多个证券

**典型 API**：`query_history_k_data_plus(code, start_date, end_date, fields, ...)`

**配置示例**：
```
TableInfo:
  - DefaultArgs: {"IDAdj":"前缀","DTFmt":"%Y-%m-%d"}

ArgInfo:
  - code: FieldType=ID
  - start_date: FieldType=StartDate
  - end_date: FieldType=EndDate
  - fields: FieldType=Field
  - frequency: FieldType=QSArg
  - adjustflag: FieldType=QSArg
```

**代码实现**：`QuantStudio/Factor/BaoStockDB.py` 第 208-257 行

### 3. QuarterTable - 季度查询

**适用场景**：API 按年份+季度查询数据

**典型 API**：`query_profit_data(code, year, quarter)`

**配置示例**：
```
TableInfo:
  - DefaultArgs: {"IDAdj":"前缀","DTFmt":"%Y-%m-%d"}

ArgInfo:
  - code: FieldType=ID
  - year: FieldType=Year
  - quarter: FieldType=Quarter
```

**代码实现**：`QuantStudio/Factor/BaoStockDB.py` 第 260-303 行

### 4. YearTable - 年度查询

**适用场景**：API 仅按年份查询数据（无季度参数）

**典型 API**：`query_dividend_data(code, year, yearType)`

**配置示例**：
```
TableInfo:
  - DefaultArgs: {"IDAdj":"前缀","DTFmt":"%Y-%m-%d"}

ArgInfo:
  - code: FieldType=ID
  - year: FieldType=Year
  - yearType: FieldType=QSArg
```

**代码实现**：`QuantStudio/Factor/BaoStockDB.py` 第 306-345 行

## 添加新表类型

如果现有表类型不满足需求，按以下步骤添加新表类型：

1. 在 `BaoStockDB.py` 中创建新类，继承 `_BSTable`
2. 定义 `__QS_ArgClass__`，设置 `TableType` 字段
3. 实现 `__QS_prepareRawData__` 方法
4. 实现 `__QS_calcData__` 方法（通常直接复用模板代码）

```python
class _NewTableType(_BSTable):
    """BaoStockDB 库中基于 XXX API 的因子表"""

    class __QS_ArgClass__(_BSTable.__QS_ArgClass__):
        TableType: Literal["NewTableType"] = Field(default="NewTableType", title="因子表类型", frozen=True)

    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        # 实现数据获取逻辑
        pass

    def __QS_calcData__(self, raw_data, factor_names, ids, dts):
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType")
        Args = self._QSArgs.to_dict(repr=False)
        ErrorFmt = {"DuplicatedIndex": "%s 的表 %s 无法保证唯一性 : {Error}, 可以尝试将 '多重映射' 参数取值调整为 True" % (self._FactorDB.Name, self.Name)}
        return _QS_calcData_WideTable(raw_data, factor_names, ids, dts, DataType, args=Args, logger=self._QS_Logger, error_fmt=ErrorFmt)
```
