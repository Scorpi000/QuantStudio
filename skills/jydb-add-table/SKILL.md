---
name: jydb-add-table
description: >
  向 QuantStudio 的聚源数据库配置文件中添加新表。当用户要求"添加一张表到 JYDB"、
  "把聚源某张表接入 QuantStudio"、"在 JYDBInfo 里配置新表"、"JYDB 读取不到某张表"、
  或需要修改 QuantStudio/Resource/JYDBInfo.xlsx 的 TableInfo / FactorInfo 工作表时触发。
  也适用于排查已配置表的字段类型、映射或表类型问题。
compatibility: Python 3.12+, QuantStudio v2.0.0, 聚源数据库, jy_doc MCP 服务（提供中文表名/字段名/数据类型，无法用数据库 introspection 替代）
---

# 向 JYDB 配置添加新表

QuantStudio 的 `JYDB` 因子库由 `QuantStudio/Resource/JYDBInfo.xlsx` 驱动。该文件决定
"聚源数据库里的哪些表、哪些字段可以被 QuantStudio 当作因子读取"，以及每张表的读取语义。

本 Skill 指导把一个聚源新表接入该配置文件的完整流程。

## 配置文件结构

`JYDBInfo.xlsx` 含 4 个工作表，**全部沿用聚源原始表结构**，只有 `TableInfo` 的
`TableClass` / `DefaultArgs` 两列是 QuantStudio 特有的语义标注。

| 工作表 | 作用 | 索引 |
|--------|------|------|
| `TableInfo` | 一张表一行，声明表级配置 | `TableName` |
| `FactorInfo` | 一个字段一行，声明字段级配置 | `(TableName, FieldName)` |
| `ExchangeInfo` | 交易所代码 → ID 后缀映射 | `ExchangeCode` |
| `SecurityInfo` | 证券类别代码 → ID 后缀映射 | `SecurityCategoryCode` |

新增一张表通常只需修改 `TableInfo` 和 `FactorInfo`；只有当表涉及新的交易所或证券类别
时，才需要动 `ExchangeInfo` / `SecurityInfo`。

两张工作表都按上述索引列去重：`TableName` 不可重复，`(TableName, FieldName)` 组合
不可重复。**重复不会报错**，但取数时会命中错误行——追加行前先确认表名/字段名不存在。

## 工作流程

### 步骤 1 — 从 MCP 查询聚源表结构

用 `jy_doc` MCP 工具获取目标表的中文表名、物理表名、字段列表和字段说明：

```
mcp__jy_doc__browse_categories()                          # 先看有哪些库
mcp__jy_doc__search_tables(keyword="日行情", category="聚源新版数据库")
mcp__jy_doc__get_table_detail(table_id=5, format="markdown")
mcp__jy_doc__get_database_page(database="接口数据库")      # 已知库、想浏览全部表时用
```

| 工具 | 返回 | 用途 |
|------|------|------|
| `browse_categories` | 有哪些库、各库表数量 | 不知道目标表在哪个库时先看全景 |
| `search_tables` | `table_id`、中文表名、物理表名、文档路径 | 按中文业务词定位表 |
| `get_table_detail` | 表说明、**字段列表**（字段名/中文名/数据类型/可空/说明）、唯一索引 | 步骤 3、4 填 `DBFieldName` 与 `FieldName` 的依据 |
| `get_database_page` | 某库下所有表的名称与 `table_id` | 浏览整库 |
| `search_online` | 同 `search_tables`，更实时 | `search_tables` 结果不足时补充 |

`get_table_detail` 是最关键的一个：**只有它给出字段级信息**，且其"唯一索引"一节
正是步骤 2 判断 `TableClass` 的直接依据。`format` 取 `markdown`（表格，字段多时更易读）
或 `text`（默认）。

`search_tables` 只索引表名/路径/描述，**不索引字段名**。要找"哪个表含某字段"，
先用中文业务词搜索，再逐张 `get_table_detail` 核对字段。

**若 MCP 不可用**：立即停止并向用户报告，让用户修复 MCP 服务后再继续。
不要绕开 MCP 直接查数据库来推进任务——`FieldName`（字段中文名）等聚源字典信息
只有 MCP 能提供，靠 `information_schema` 之类的 introspection 拿不到，
自行拼凑会产出错误的配置。报告时说明失败的工具与现象，便于用户排查。

### 步骤 2 — 判断表的读取语义（选 TableClass）

这是最关键的一步。`TableClass` 决定 QuantStudio 如何把数据库行翻译成"时点 × ID × 因子"
三维数据。**表类型选错会导致数据静默错误**，务必按下表核对聚源表的实际主键结构。

| TableClass | 数据形态 | 判据 | 依据 |
|------------|----------|------|------|
| `WideTable` | 一个 ID 字段 + 一个时点字段，其余字段是因子 | 表内同时有证券 ID 和日期，且 (ID, 日期) 唯一 | `SQL_WideTable` |
| `FeatureTable` | 只有 ID、无时点，是一张截面快照 | 表内无日期字段，数据是"当前状态" | `SQL_FeatureTable` |
| `MappingTable` | 一个 ID + 起始时点 + 结束时点，期间内取值有效 | 有 `InDate`/`OutDate` 成对字段（如成分、评级、担保关系） | `SQL_MappingTable` |
| `FinancialTable` | 财报：ID + 报告期 + 公告日 | 三大报表、财务指标类表 | `SQL_FinancialTable` |
| `NarrowTable` | 一行一个 (ID, 时点, 因子名, 因子值) | 长表格式，因子名本身是一个字段的取值 | `SQL_NarrowTable` |
| `MacroTable` | 宏观：ID + 截止日 + 发布日 | 宏观指标数据 | `SQL_MacroTable` |
| `ConstituentTable` | 成分表：ID 在某类别中、有纳入/剔除日 | 有类别字段 + 纳入日 + 剔除日，因子值是 0-1 | `SQL_ConstituentTable` |
| `TimeSeriesTable` | 只有时点、无 ID，如利率、宏观指数单值序列 | 表内无证券 ID 字段 | `SQL_TimeSeriesTable` |
| `FinancialIndicatorTable` | 财务指标表（报告期无公告日，需回补） | 仅用于 A股/公募基金的指标表 | `JYDB._FinancialIndicatorTable` |
| `AnalystConsensusTable` | 分析师汇总预测 | 含 FY0/FY1 报告期列 | `JYDB._AnalystConsensusTable` |
| `AnalystEstDetailTable` | 分析师盈利预测明细 | 需自定义算子聚合 | `JYDB._AnalystEstDetailTable` |
| `AnalystRatingDetailTable` | 分析师评级明细 | 需自定义算子聚合 | `JYDB._AnalystRatingDetailTable` |

**选择顺序参考**：`WideTable` 是最常用的类型，其次是 `FeatureTable`、`MappingTable`；
财务、宏观、分析师、成分等专用类型较少见，但一旦表属于这些形态就必须用对应类型。

不确定时优先选 `WideTable`（最通用），但它要求 (ID, 时点) 能唯一确定一行——
若聚源表在单日单 ID 处有多行，应改用 `NarrowTable` 或 `MappingTable`。
`get_table_detail` 返回的**唯一索引**就是判据：唯一索引为 `(ID字段, 时点字段)`
即 `WideTable`；含 `InDate`/`OutDate` 即 `MappingTable`；二者都没有再考虑其余类型。

### 步骤 3 — 填写 TableInfo 行

在 `TableInfo` 工作表追加一行，按下表逐列填写。**列名必须完全一致**。
`{DBTable}` 与 `{MainTable}` 是代码在运行时替换的占位符
（见 `QuantStudio/Factor/FactorUtils.py:480`）。

| 列名 | 必填 | 含义与填法 |
|------|------|------------|
| `TableName` | 是 | **QuantStudio 内部表名，可自由命名**（不要求等于聚源中文表名）。`getTable(name)` / `TableNames` 用的是这一列，与聚源侧无关 |
| `DBTableName` | 是 | 聚源**物理表名**，如 `QT_DailyQuote`；`TableClass` 非空时它必须真实存在于数据库中（见下） |
| `TableClass` | 是 | 步骤 2 选定的类型；留空则该表不出现在 `TableNames` 中 |
| `DefaultArgs` | 否 | Python dict 字面量（用 `eval` 解析，故只能写字面量），作为该表默认参数，会被 `getTable(args=...)` 覆盖。常用 `{'MultiMapping':False}` |
| `MainTableName` | 否 | 主表（通常是 `SecuMain` 证券主表）。用于把 ID 解析成带后缀的证券代码 |
| `MainTableID` | 否 | 主表中与 ID 关联的字段，如 `SecuCode` |
| `JoinCondition` | 否 | 主表连接条件，如 `{DBTable}.InnerCode={MainTable}.InnerCode` |
| `MainTableCondition` | 否 | 主表的过滤条件，如 `{MainTable}.SecuCategory IN (1,41) AND {MainTable}.SecuMarket IN (83,90,18)` |
| `SecurityCategory` | 否 | 证券类别 → 后缀，格式 `字段:代码1,代码2`，如 `SecuCategory:1,41`。查 `SecurityInfo` 得到后缀 |
| `Exchange` | 否 | 交易所 → 后缀，格式 `字段:代码1,代码2`，如 `SecuMarket:83,90,18`。查 `ExchangeInfo` 得到后缀 |
| `DefaultSuffix` | 否 | 无匹配时的默认后缀，如 `.OF`。留空表示不加后缀 |
| `SecurityType` | 否 | 证券类型说明，如 `A股`、`公募基金`。仅 `FinancialIndicatorTable` **必填**且当前只接受 `A股`/`公募基金`，其余类型填了不生效 |
| `Description` | 否 | 表说明 |

**`TableName` 可任意命名，且允许同一 `DBTableName` 出现多行**。这是常规做法而非错误：
同一物理表以不同 `TableName` + 不同 `DefaultArgs` / 后缀配置注册两次，就得到两套读取语义。
例如 `QT_AdjustingFactor` 可同时注册为 `复权因子表` 与 `公募基金行情复权因子表`，
后者用 `DefaultSuffix=.OF` 区分基金 ID——配置中这类成对注册（A股版 + `.OF` 基金版）
很常见。**表名唯一性约束在 `TableName` 上**，与 `DBTableName` 无关。

**`TableClass` 非空时整行会被校验**：`TableNames` 只纳入 `DBTableName` 真实存在于数据库的表
（`JYDB.py:975` 用 `getDBTable()` 取全部物理表名后过滤）。写错物理表名（含多打一个后缀、
库版本不匹配）不会报错，只会让该表**静默地从 `TableNames` 里消失**——步骤 6 的
"表已注册"检查就是为抓这个。`TableClass` 留空的行则不参与此校验。

关于后缀解析（`QuantStudio/Factor/FactorUtils.py:562`）：ID 最终形态是
`CONCAT(SecuCode, 后缀)`。后缀按 `SecurityCategory` → `Exchange` → `DefaultSuffix` 的
顺序层层包裹成 `CASE WHEN` 表达式。**三者可叠加**，`DefaultSuffix` 是最内层兜底。

**绝大多数表只填 `Exchange`**——A股表通常用 `Exchange = SecuMarket:83,90,18`
即可（83=上交所、90=深交所、18=北交所）。

**`SecurityCategory` 只在需要按证券类别而非交易所区分 ID 时使用。**两者叠加时
`SecurityCategory` 在外层、`Exchange` 退化为它的 `ELSE` 分支，即"先按证券类别判、
判不出再按交易所判"。若两个条件能覆盖同一批 ID，按此顺序会以 `SecurityCategory`
的结果为准——想只要交易所后缀就**不要填** `SecurityCategory`。

### 步骤 4 — 填写 FactorInfo 行

为该表的**每个需要暴露的字段**追加一行。不需要全部字段都配置——
只配置用户实际要用的字段即可，未配置的字段在 QuantStudio 中不可见。

| 列名 | 必填 | 含义与填法 |
|------|------|------------|
| `TableName` | 是 | 与 TableInfo 一致 |
| `DBFieldName` | 是 | 聚源**物理字段名**，如 `ClosePrice` |
| `FieldName` | 是 | 字段中文名（因子名），QuantStudio 用 `getFactor("收盘价(元)")` 访问 |
| `DataType` | 是 | **聚源原始类型字符串**，如 `decimal(10,4)`、`int`、`datetime`、`varchar(300)`。QS 据此推断 double/string/object，**不要改写为 QS 类型** |
| `FieldType` | 见下 | 字段角色标注，决定它是否成为因子 |
| `Supplementary` | 否 | 语义随 `FieldType` 变化，见下表 |
| `Description` | 否 | 字段说明 |
| `RelatedSQL` | 否 | 取值映射 SQL 或 dict 字面量，见"取值映射" |

`FieldType` 取值：

| FieldType | 含义 | `Supplementary` 的用途 |
|-----------|------|------------------------|
| `因子` | 普通因子字段（最常用） | 一般留空；若含 `从表` 则做左连接，见下 |
| `ID` | ID 字段（每表至多一个） | 留空 |
| `Date` | 时点字段 | 填 `Default` 表示默认时点字段（多时点表中用于指定一个表级默认） |
| `AnnDate` | 公告日字段 | 留空（财务/宏观表使用） |
| `EndDate` | 结束时点字段 | 填 `不包含` 表示结束时点当日**不再**有效；留空表示包含（默认） |
| `ReportDate` | 报告期字段 | 留空（财务表使用） |
| `Condition` | 固定筛选条件 | 条件值，如 `20`；多值用逗号分隔。会生成 `字段 IN (...)` 附加到 WHERE |
| `AdjustType` | 调整类型字段 | 取值集合，如 `2,1`，由 `SQL_FinancialTable` 生成 `字段 IN (2,1)`（财务表使用） |
| `Value` | 窄表的因子值字段 | 留空 |
| `Factor` | 窄表的因子名字段 | 留空 |
| `Group` | 成分表的类别字段 | 类别映射 SQL 或 dict |
| `CurSign` | 成分表当前状态字段 | 留空 |
| `Period` | 分析师表的周期字段 | 留空 |
| `Institute` / `Analyst` | 分析师表辅助字段 | 留空 |

`FieldType` 留空（NaN）的字段**不会成为因子**，仅作内部辅助（如系统常量表各字段）。

**`Date` / `AdjustType` / `Condition` 多个字段同时存在时**：`Date` 用 `Supplementary=Default`
指定表级默认时点；`AdjustType` 由 `SQL_FinancialTable` 通过生成的 `AdjustType` 参数挑选
（默认取第一个 `AdjustType` 字段）；`Condition` 则**所有** `Condition` 字段都会生效、
条件 AND 叠加。

#### `FieldName` 中禁止出现 `/`

写 `FieldName` 时**必须把聚源中文名里的 `/` 替换成 `-`**。例如聚源的
`收盘价(元/股)` 应写成 `收盘价(元-股)`，`净利润/营业总收入(%)` 写成
`净利润-营业总收入(%)`。

原因：`FactorTable.getFactor` 会把因子名直接作为节点的 `Name`
（`QuantStudio/Factor/FactorTable.py:83`），而节点路径由 `/` 拼接
（`QuantStudio/Core/TreeEngine.py` 中形如 `"/".join(iPath)` 的写法）。
因子名里的 `/` 会被当成路径分隔符，导致节点路径解析错乱。
**代码中没有任何校验会拦住它**，错误只会在回测或树形引擎运行时才暴露，
且现象是路径问题而非因子问题，很难定位。

> 历史遗留：现有 `JYDBInfo.xlsx` 中仍有一批未替换 `/` 的字段名，属待修复的配置，
> 不代表正确做法。新增字段一律按上述规则写 `-`。

**ID 字段的 `DataType` 决定 ID 是否加引号**：数值型 ID（如 `InnerCode` 为 `int`）会被
`CAST` 成字符串再与主表代码拼接。选错类型会导致 SQL 语法错误或 ID 不匹配。
ID 字段绝大多数是数值型（`int`/`bigint`/`number`），只有少数是字符串型，
所以**默认应当填聚源给出的数值类型**，除非聚源的 ID 字段确实是 `varchar`。

**跨表取字段**（`Supplementary` 含 `从表`）：格式
`从表:<从表名>:<连接字段>` 或 `从表:<从表名>:<连接字段>:<TypeCode>`。
用于从另一张表借字段，代码会生成 `LEFT JOIN`（见 `QuantStudio/Factor/JYDB.py:95`）。
仅在字段确实不在本表时才用；能用主表关联解决的不要用这个。

该机制用于聚源把明细存在 `*_SE` 附表里、用 `TypeCode` 区分类型的场景
（如 `从表:LC_Dividend_SE:ID:1`）。注意此时 `DBFieldName` 写的是**从表**中的字段名。

### 步骤 5 — 取值映射（仅当需要）

聚源常以代码存分类信息（如 `SecuMarket=83` 表示上交所）。QuantStudio 通过
`RelatedSQL` 或 `Supplementary` 把这些代码翻译成可读值。两种写法：

**dict 字面量**（枚举固定时用）：
```
RelatedSQL = {1:'是',2:'否'}
```
字段名以 `_R` 结尾表示这是映射后的衍生字段，如 `IfTradingDay_R`。

**SQL 查询**（取值需从库中动态取时用）：
```
RelatedSQL = SELECT DM, MS FROM {TablePrefix}CT_SystemConst WHERE LB=201 AND DM IN ({Keys})
```
可用占位符（见 `QuantStudio/Factor/JYDB.py:132` 的 `_QS_getValueMapping`）：

| 占位符 | 替换为 |
|--------|--------|
| `{TablePrefix}` | 表前缀（`TablePrefix` 参数） |
| `{Keys}` | 候选键集合。**取值来自 `values` 参数**，`values=None` 时退化为"从本表 DISTINCT 取该字段"；字符串字段会加引号，数值字段不加 |
| `{KeyCondition}` | 键条件，用法为 `{KeyCondition}<字段>`；同样依赖 `values`，`None` 时为 `字段 IS NOT NULL` |
| `{SecuCode}` | 证券主表 ID 表达式，由 `_getSecuMainIDField()` 按 `ExchangeInfo` 拼出的 `CASE SecuMarket ... CONCAT(SecuCode, 后缀)` |

查询结果按"**第一列 = 原值，第二列 = 映射值**"取（代码里是 `{jVal: jRelatedVal}`）。

映射后的值类型由**原字段**的 `DataType` 决定（代码对 `_R` 结尾的字段会去掉后缀再查，
见 `JYDB.py:142`），所以加了 `_R` 的字段，元数据要写 `varchar` 之类的字符串类型。

### 步骤 6 — 验证

配置改完后**必须实际跑一遍**，仅看 Excel 无法发现类型/ID 错误。

**先做离线检查**（不需要数据库，几秒完成）——能立刻发现表名重复、索引错乱、
列缺失这类配置级错误：

```python
import os, logging, sys; sys.stdout.reconfigure(encoding='utf-8')
from QuantStudio import __QS_MainPath__
from QuantStudio.Factor.JYDB import _importInfo
XLSX = os.path.join(__QS_MainPath__, 'Resource', 'JYDBInfo.xlsx')   # 不依赖当前工作目录
TI, FI, EI, SI = _importInfo(None, XLSX, logging.getLogger('t'))
print('TableInfo:', TI.shape, TI.index.name)
print('FactorInfo:', FI.shape, FI.index.names)
print('新表已入库:', '你的表名' in TI.index)
print('新表字段数:', FI.loc['你的表名'].shape[0])
print('TableName 重复:', int(TI.index.duplicated().sum()))          # 应为 0
```

用 `__QS_MainPath__`（即 `QuantStudio` 包目录）拼路径，**不要写
`'QuantStudio/Resource/JYDBInfo.xlsx'` 这类相对路径**——后者要求恰好从仓库根目录运行，
换工作目录就会 `FileNotFoundError`。

`_importInfo` 会按 `TableName` 对 TableInfo 建索引、按 `(TableName, FieldName)` 对
FactorInfo 建索引。**若出现表名或字段名重复，pandas 不会报错**，但后续按表名取数据时
会取到错误行。离线检查中确认 shape 与索引名正常、重复数为 0，即说明配置结构无误。

> `_importInfo` 与 `JYDB._initInfo` 走的是两条不同的路径：`_initInfo` 只在传入
> `DBInfoFile` 参数时才调 `_importInfo`，否则走 `_updateInfo`——**只要 hdf5 缓存
> 比 xlsx 新就直接读缓存，完全不解析 xlsx**。所以这个离线检查是绕过缓存、直接校验
> 你刚编辑的 xlsx 的唯一手段。

**再跑在线验证**（需连聚源库）：

```python
from QuantStudio.Factor.JYDB import JYDB
SDB = JYDB().connect()
print('表已注册:', '你的表名' in SDB.TableNames)
FT = SDB.getTable('你的表名')
print('因子数:', len(FT.FactorNames))
print('因子样例:', FT.FactorNames[:10])
print('ID 样例:', FT.getID()[:5])
print('时点样例:', FT.getDateTime()[:5])
print(FT.readData(factor_names=FT.FactorNames[:2], ids=FT.getID()[:3], dts=FT.getDateTime()[:3]))
```

以上代码用项目约定的 Python 解释器运行（见项目的 `CLAUDE.local.md`）即可，
**不要求特定工作目录**（路径都由 `__QS_MainPath__` / `__QS_ConfigPath__` 解析）；
唯一的例外是，若当前目录下恰好存在 `QuantStudio` 子包，`import QuantStudio`
会优先命中它而不是已安装的版本。

验证要点（按重要性排序）：

1. **ID 形态正确**：`getID()` 返回的应是带后缀的证券代码（如 `600000.SH`），
   不是裸数字或内部编码。裸编码说明 `MainTableName`/`JoinCondition` 配错
2. **因子名正确**：`FactorNames` 应列出配置的中文因子名
3. **时点正确**：`getDateTime()` 非空且落在合理区间
4. **数据非空且有值**：`readData` 返回的 Panel 应有真实数值，不是全 NaN。
   全 NaN 通常是 `Condition` 条件写错或时点/ID 字段选错
5. **数值类型正确**：数值因子在 Panel 中应为 float，字符串字段为 object

## 注意事项

### 信息文件缓存

`_updateInfo` 会比较 `JYDBInfo.xlsx` 与 `JYDBInfo.hdf5` 的修改时间
（`QuantStudio/Factor/JYDB.py:57`）。**只要 xlsx 比 hdf5 新就会重新解析并覆盖缓存**，
所以正常编辑保存后**无需手动删除 hdf5**；仅当 xlsx 修改时间反而更早
（少见，但某些编辑器/复制文件会保留原时间戳）时才需要手动删除
`QuantStudio/Resource/JYDBInfo.hdf5`（该文件在 `.gitignore` 中）强制重建。

**注意缓存命中时不会做任何表名校验**——若 xlsx 比 hdf5 旧，你新加的表根本不会被解析，
而步骤 6 的离线检查（直接调 `_importInfo`）仍会显示"已入库"，两处结论相反即可判定
是缓存问题。

若 xlsx 解析抛异常，`_updateInfo` 会回退到旧的 hdf5 缓存，症状是
"改了配置但没生效"。此时看日志中是否有 "更新数据库信息文件 ... 失败" 警告
（`writeNestedDict2HDF5` 写缓存失败会打这条，注意它报的是**写**缓存失败，
与上面的读缓存命中是两回事）。

### 不要臆造字段信息

`FieldName`（中文名）和 `DataType`（原始类型）必须来自 `get_table_detail` 或用户确认。
凭空编造的字段名会让 `getFactor()` 找不到因子，而错误的 `DataType` 会导致
数值被当成字符串（或反之），这类错误不会报错，只会静默产出错误数据。

### 环境无关性

本 Skill 不写死表名、字段名、表前缀和数据库连接信息——这些都通过 `jy_doc` MCP
或用户输入动态获取。聚源库可能存在多个版本（不同 `TablePrefix`），
配置时以用户实际连接的那个库为准。判断某张表该怎么配时，始终以 `get_table_detail`
返回的实际结构和现有配置为准，不要依赖任何记下来的统计或清单。

## 相关参考

- `QuantStudio/Factor/FactorUtils.py` — 各 `SQL_*Table` 基类，是 TableClass 语义的权威定义
  （`SQL_WideTable:764`、`SQL_NarrowTable:1156`、`SQL_FeatureTable:1393`、
  `SQL_TimeSeriesTable:1476`、`SQL_MappingTable:1718`、`SQL_ConstituentTable:1953`、
  `SQL_FinancialTable:2211`、`SQL_MacroTable:2457`、`SQL_Table:393`）
- `QuantStudio/Factor/JYDB.py` — 聚源专用表类（`_WideTable:224` 起、
  `_FinancialIndicatorTable:306`、`_AnalystConsensusTable:474`、`_AnalystEstDetailTable:658`、
  `_AnalystRatingDetailTable:818`）、ID 解析（`_getSecuMainIDField:124`）、
  取值映射（`_QS_getValueMapping:132`）、从表左连接（`_genFieldSQLStr:95`）、
  信息文件导入（`_importInfo:23` / `_updateInfo:57`）

配置改完后如何用 `getTable()` / `getFactor()` 取数、以及 `Table` / `Factor` 的完整 API，
见 QuantStudio 框架自身的 Skill（若已安装）。
