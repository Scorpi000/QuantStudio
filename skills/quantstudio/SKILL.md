---
name: quantstudio
description: >
  Use this skill when writing QuantStudio (v2.0.0) code in this project — factor development,
  strategy backtesting, risk modeling, portfolio optimization, or anything involving the
  QuantStudio computation graph engine. The framework is a Chinese A-share quantitative finance
  research platform. This skill covers all modules: Core (QS_Object/QS_Args patterns, Panel,
  Node/Engine computation graph), Factor (JYDB, HDF5DB, factor operators, data reading, defFactor
  template), BackTest (strategy, section factor tests, performance attribution), Risk (Barra
  multi-factor models, HDF5 risk DB), and PortfolioConstructor (mean-variance, risk budgeting,
  constraints, CVXPC solver). ALWAYS use this skill when the user asks to write QuantStudio code,
  develop factors, run backtests, build risk models, or do portfolio optimization in this repo.
  For factor development: use MCP tools to locate database tables/fields (never hardcode DB
  internals), then follow the defFactor() template. Code location and directory structure should
  follow the user's current task instructions, not conventions stored here.
---

# QuantStudio 框架使用指南

## 环境

- Python: conda 环境 `QS312`，路径 `D:\miniforge\envs\QS312`
- 配置目录: `C:\Users\lenovo\QuantStudioConfig\`（通过 `from QuantStudio import __QS_ConfigPath__` 获取）
- 聚源数据库: PostgreSQL，连接信息在 `~/QuantStudioConfig/JYDBConfig.json`

## 核心约定

### QS_Object 和 QS_Args

所有 QuantStudio 对象继承自 `__QS_Object__`，每个对象包含一个 `__QS_Args__` 参数集。

```python
from QuantStudio.Core import __QS_Object__, __QS_Args__
```

**构造方式**: `obj = SomeClass(args={...}, config_file=None, logger=...)`

参数优先级: **显式 args > JSON config_file > Pydantic 字段默认值**

配置文件默认从 `~/QuantStudioConfig/` 加载，只需给文件名（不含路径）。

**关键属性**:
- `obj.QSID` — 参数的确定性 SHA256 哈希，相同 QSID = 相同行为，用于缓存/去重
- `obj.Args` — 参数集对象，可用 `obj.Args["key"] = val` 修改，用 `obj.Args.to_dict()` 导出
- `obj.Logger` — 日志对象

**Args 元信息**: `obj.Args.meta(key)` 查看参数的 annotation/frozen/exclude/repr 等元信息；`obj.Args.info()` 查看可读说明。

**创建变体**: `new_obj = obj.new(args={"Name": "new_name"})` — 未指定的参数沿用原对象的值。

### QSID 和缓存

QSID 相同的两个对象产生相同结果。框架用此实现计算缓存。`exclude=True` 的参数不参与 QSID 生成；`frozen=True` 的参数初始化后不可修改。

### 日志

```python
from QuantStudio.Core import setDefaultLogLevel
import logging
setDefaultLogLevel(level=logging.WARNING)
```

### Panel 数据结构

`Panel` 是三维带标签数组（items × major_axis × minor_axis），类似 pandas 早期 Panel。

```python
from QuantStudio.Core.QSObject import Panel
p = Panel(data=np.random.rand(2, 4, 3), items=["close", "open"],
          major_axis=DTs, minor_axis=IDs)
p.loc["close", dt1:dt2, ["000001.SZ", "000002.SZ"]]  # 标签索引
p.iloc[0, 0:3, [0, 2]]                                 # 位置索引
p.to_frame(filter_observations=False)                    # 转 MultiIndex DataFrame
p.values                                                 # 转 numpy array
```

**因子数据规范**: Panel 的 items=因子名, major_axis=时点, minor_axis=证券代码。所有因子 API 遵循此约定。

### 计算图框架

```python
from QuantStudio.Core.Node import Node, Context, DTLocalContext, DTInitData
from QuantStudio.Core.CalcEngine import Engine
```

每个计算节点继承 `Node`，必须实现 `backward_compute`（主计算逻辑）：

```python
class MyNode(Node):
    def __init__(self, deps=[], args={}, **kwargs):
        super().__init__(deps=deps, args=args, **kwargs)
    def backward_compute(self, path, bwd_data_list, context, local_context=None):
        # bwd_data_list 是从子节点传来的数据
        return result
```

**引擎执行**:
```python
engine = Engine()
results = engine.run(node_list, context, init_data_list=[...], fwd_data_list=[...])
```

## 因子框架

### 因子开发速查（新因子开发必读）

#### 开发流程（按顺序执行）

1. **定位数据源** — 用 MCP 工具搜索数据库表和字段（数据库结构和文档完全通过 MCP 查询，不要硬编码表名或字段名）
2. **检查字段质量** — 关注 `有值率(%)`，低于 30% 的字段可能覆盖不足
3. **查看 QS 用法** — 用 `query_qs_get_factor_help` 获取该表在 QuantStudio 中的 `getTable` 参数和 `getFactor` 示例
4. **编写因子代码** — 按 `defFactor()` 模板（见下）构建因子，存放路径和目录结构遵循用户当前任务的指示
5. **运行测试** — 用 `if __name__=="__main__"` 块验证，先小样本再全量

#### 财务表 CalcType 参数

`CalcType` 是 QuantStudio 框架级参数，适用于所有财务类因子表：

- `"最新"` — 每个时点取当时已公告的最新财报值。同一财报期值在不同交易日保持不变，直到新财报公告
- `"单季度"` — 单季度数据（需两期报表差分）
- `"TTM"` — 滚动四季度合计

`ReportDate` 筛选: `"所有"` | `"定期报告"` | `"年报"` | `"中报"` | `"一季报"` | `"三季报"`

其他表参数（如 `FilterCondition`、`LookBack` 等）通过 `query_qs_get_factor_help` 查询获取。

#### defFactor() 模板

```python
from typing import List
from QuantStudio.Factor.Factor import Factor
from QuantStudio.Factor.JYDB import JYDB
from QuantStudio.Factor.BasicOperator import rename

def defFactor() -> List[Factor]:
    SDB = JYDB().connect()
    FT = SDB.getTable("<表名>", args={"CalcType": "最新"})
    Factor1 = FT.getFactor("<因子名1>")
    Factor2 = FT.getFactor("<因子名2>")
    Factor3 = rename(Factor1 / Factor2, factor_name="DerivedFactor")
    return [Factor1, Factor3]

if __name__ == "__main__":
    import datetime as dt
    SDB = JYDB().connect()
    DTs = SDB.getTradeDay(dt.datetime(2025, 1, 1), dt.datetime(2025, 12, 31))
    IDs = SDB.getStockID(date=dt.datetime(2025, 6, 30))
    factors = defFactor()
    for f in factors:
        print(f"{f.Name}: QSID={f.QSID}")
    data = factors[0].readData(ids=IDs[:5], dts=DTs[-5:])
    print(data)
```

#### 因子命名

- 从 JYDB 直接取的因子：使用中文名 `FT.getFactor("商誉")`
- 衍生计算产生的新因子：使用英文名 `rename(..., factor_name="GoodwillToAsset")`
- 因子返回值按 `[基础因子, 衍生因子]` 排列

### 导入

```python
from QuantStudio.Factor.api import (DataFactor, FactorContext, FactorInitData,
    FactorLocalContext, HDF5DB, JYDB, BaoStockDB,
    makeFactorOperator, FactorOperatorized,
    PointOperation, SectionOperation, TimeOperation, PanelOperation,
    rename, FeatherDTCache, FeatherFactorCache)
import QuantStudio.Factor.FactorOperator as fo  # 内置算子
```

### 因子数据模型（三层结构）

```
因子库 (FactorDB) → 因子表 (FactorTable) → 因子 (Factor)
```

- 因子表数据 = Panel(items=因子名, major_axis=时点, minor_axis=证券代码)
- 因子数据 = DataFrame(index=时点, columns=证券代码)

因子分两类:
- **基础因子**: 直接从原始数据转化，通过 `FT.getFactor("名称")` 获取或用 `DataFactor` 构造
- **衍生因子**: 通过算子作用于其他因子产生，形成计算图 DAG

### JYDB（聚源数据库）

```python
from QuantStudio.Factor.JYDB import JYDB

SDB = JYDB().connect()                # 从 JYDBConfig.json 读取连接信息
SDB.TableNames[:5]                    # 查看可用表
FT = SDB.getTable("日行情表", args={"LookBack": 0})  # 获取因子表
FT.FactorNames                        # 查看表中因子
F = FT.getFactor("收盘价(元)")        # 获取因子对象

# 读取数据
F.readData(ids=["000001.SZ", "000002.SZ"], dts=DTs)
FT.readData(factor_names=["收盘价(元)", "今开盘(元)"], ids=IDs, dts=DTs)
```

**JYDB 辅助方法**:
- `SDB.getTradeDay(start_date, end_date, exchange="SSE")` — 获取交易日
- `SDB.getStockID(exchange=("SSE","SZSE","BSE"), date=None, is_current=True)` — 获取股票代码
- `SDB.getMutualFundID(...)` / `SDB.getFutureID(...)` / `SDB.getOptionID(...)` — 获取基金/期货/期权代码

**因子表类型**（通过 `TableType` 参数指定）:
- `WideTable` — 宽表：一个字段标识 ID，一个字段标识时点，其余字段为因子。参数: `DTField`（时点字段）、`LookBack`（缺失回溯天数，inf=无限回溯）、`PublDTField`（公告时点字段）、`MultiMapping`（高维数据）、`Operator`（合并函数）
- `FeatureTable` — 特征表：无时点维度或忽略时点
- `FinancialTable` — 财务表：参数 `CalcType`（"最新"/"单季度"/"TTM"）、`ReportDate`（报告期筛选）、`YearLookBack`/`PeriodLookBack`（回溯年数/期数）、`PublDTField`（公告时点）
- `MappingTable` — 映射表：`DTField` 起始时点 + `EndDTField` 结束时点，区间内填充相同值
- `ConstituentTable` — 成份表：`GroupField` 类别字段，输出 0/1 二值
- `TimeSeriesTable` / `NarrowTable` — 其他格式

### HDF5DB（本地 HDF5 因子库）

```python
from QuantStudio.Factor.HDF5DB import HDF5DB
FDB = HDF5DB(args={"MainDir": "./data/HDF5"}).connect()
FT = FDB.getTable("stock_cn_day_bar")
F = FT.getFactor("close")
data = F.readData(ids=IDs, dts=DTs)
```

### 因子运算

四种运算类型:

| 运算 | 类 | 参数 | 说明 |
|------|------|------|------|
| 单点运算 | `PointOperator` | DTMode, IDMode, Arity | 同时点、同证券 |
| 时序运算 | `TimeOperator` | DTMode, IDMode, LookBack, StartDT, iInitFactor | 同证券、历史序列 |
| 截面运算 | `SectionOperator` | DTMode, Arity | 同时点、其他证券 |
| 面板运算 | `PanelOperator` | DTMode, LookBack, StartDT, iInitFactor | 历史序列 + 截面 |

#### 创建算子

**方式1: `makeFactorOperator` 工厂函数（推荐）**
```python
from QuantStudio.Factor.FactorOperation import makeFactorOperator

def my_func(f, idt, iid, x, args):
    # f: 因子对象, idt: 当前时点, iid: 当前ID, x: 描述子数据list, args: 附加参数
    return (x[0] + x[1]) / 2

calcMid = makeFactorOperator(my_func, operator_type="Point",
    args={"Name": "calcMid", "Arity": 2, "DTMode": "多时点", "IDMode": "多ID"})
Mid = calcMid(High, Low, factor_args={"Name": "Mid"})
```

**方式2: `FactorOperatorized` 装饰器（最简洁）**
```python
from QuantStudio.Factor.FactorOperation import FactorOperatorized

@FactorOperatorized(operator_type="Section", args={"Arity": 1, "DTMode": "多时点"})
def calcZScore(f, idt, iid, x, args):
    return (x[0] - np.nanmean(x[0], axis=1, keepdims=True)) / np.nanstd(x[0], axis=1, keepdims=True)

ZScore = calcZScore(PB, factor_args={"Name": "PB_ZScore"})
```

**方式3: 直接使用 `PanelOperation` 等（不推荐，仅在面板运算时使用）**
```python
PanelFactor = PanelOperation(descriptors=[dep_factor],
    args={"Name": "MyFactor", "Operator": my_operator})
```

**方式4: 运算符重载**
```python
from QuantStudio.Factor.BasicOperator import rename
Mid = rename((High + Low) / 2, factor_name="Mid")
```

#### 算子 calculate 函数的 x 参数形状

| DTMode | IDMode | x[i] shape | 返回值 shape |
|--------|--------|-----------|-------------|
| 单时点 | 单ID | scalar | scalar |
| 单时点 | 多ID | (len(iid),) | (len(iid),) |
| 多时点 | 单ID | (len(idt),) | (len(idt),) |
| 多时点 | 多ID | (len(idt), len(iid)) | (len(idt), len(iid)) |

**时序算子**: x[i] shape 中第一维为 LookBack[i]+len(idt)（或 LookBack[i]+1），追溯 LookBack[i] 期历史。

#### 内置算子 (`QuantStudio.Factor.FactorOperator as fo`)

| 算子 | 类型 | 说明 |
|------|------|------|
| `fo.Log(base=np.e)` | Point | 对数 |
| `fo.AsType(dtype="double")` | Point | 类型转换 |
| `fo.NotNull()` | Point | 检查非空 |
| `fo.IsIn(value_set)` | Point | 判断是否在集合中 |
| `fo.ApplyArrayFunc(func)` | Point | 自定义 apply |
| `fo.Applymap(func)` | Point | 逐元素 apply |
| `fo.Where(mask)` | Point | 条件选择 |
| `fo.Fetch()` | Point | 取描述子数据 |
| `fo.Sum()` / `fo.Max()` / `fo.Min()` | Point | 多元聚合 |
| `fo.Rank(ascending=True)` | Point | 排名 |
| `fo.Mean()` / `fo.Std()` | Point | 均值/标准差 |
| `fo.Regress(y_index)` | Point | 回归 |
| `fo.RegressChangeRate(y_index)` | Point | 回归变化率 |
| `fo.ToList()` / `fo.ToCompound()` | Point | 列表/复合类型转换 |
| `fo.Lag(lag_period)` | Time | 滞后 |
| `fo.RollingRank(lookback, ascending)` | Time | 滚动排名 |
| `fo.RollingMean(lookback)` | Time | 滚动均值 |
| `fo.RollingApply(func, lookback)` | Time | 滚动 apply |
| `fo.RollingChangeRate(lookback)` | Time | 滚动变化率 |
| `fo.RollingRegress(y_index, lookback)` | Time | 滚动回归 |
| `fo.SectionRank(ascending, uniformization)` | Section | 截面排名 |
| `fo.Aggregate(aggr_func)` | Section | 截面聚合（如 np.nansum） |
| `fo.Disaggregate(aggr_func)` | Section | 截面分解 |
| `fo.ConcatSection()` | Section | 截面拼接 |
| `fo.ChgSection()` | Section | 截面切换 |
| `fo.SectionRegress(y_index)` | Section | 截面回归 |
| `fo.PanelRegress(y_index)` | Panel | 面板回归 |

使用示例:
```python
import QuantStudio.Factor.FactorOperator as fo
log_ret = fo.Log()(Close)
rank_factor = fo.SectionRank(ascending=True, uniformization=True)(EP, mask=mask)
signal = rename(rank_factor >= 0.8, factor_name="Top20Pct")
```

### DataFactor

```python
from QuantStudio.Factor.Factor import DataFactor
F = DataFactor(data=1)                           # 标量
F = DataFactor(data=pd.DataFrame(np.random.randn(3,3), index=DTs, columns=IDs))  # DataFrame
F.readData(ids=IDs, dts=DTs)
```

### 因子缓存

```python
from QuantStudio.Factor.FactorCache import FeatherFactorCache
with FeatherFactorCache(args={"DTRuler": DTRuler, "CacheDir": "./Cache",
                               "StartMode": "new"}) as Cache:
    with FactorContext(DTRuler=DTRuler, SectionIDs=SectionIDs, DataCache=Cache) as Context:
        ...
```

### 读取数据完整参数

```python
factor.readData(ids=IDs, dts=DTs, section_ids=SectionIDs, dt_ruler=DTRuler)
```

- `ids`: 目标证券代码
- `dts`: 目标时点
- `section_ids`: 截面 ID（用于截面运算的全体截面）
- `dt_ruler`: 时点标尺（用于时序运算的完整时间轴）

## 其他模块

- **[回测框架](backtest.md)**: 策略回测、自定义策略、截面因子测试(IC/分位组合/Fama-MacBeth)、Brinson 业绩归因
- **[风险模型与组合优化](risk_portfolio.md)**: Barra 多因子风险模型、均值方差/风险预算/最大化分散化优化、cvxpy 求解器

## Tools 工具模块

```python
from QuantStudio.Tools.api import (Math, DateTime, Strategy, Preprocess,
    File, Visualization, genAvailableName)
from QuantStudio.Tools.Visualization import qs_help  # 查看帮助
from QuantStudio.Tools.DateTimeFun import getMonthLastDateTime
```

- `qs_help(obj)` — 查看任何 QS 对象的帮助信息
- `getMonthLastDateTime(dt_list)` — 从时点列表中提取月末时点
- `genAvailableName(base_name, existing_names)` — 生成不重名的新名称

## 常用模式总结

> **新因子开发请先阅读上方"因子开发速查"章节**，按步骤执行。数据库表和字段通过 MCP 工具查询，不要硬编码。

### 因子开发完整流程（defFactor 模板）

```python
from typing import List
import numpy as np
import datetime as dt
from QuantStudio.Factor.Factor import Factor
from QuantStudio.Factor.JYDB import JYDB
from QuantStudio.Factor.BasicOperator import rename
from QuantStudio.Factor.FactorOperation import FactorOperatorized
import QuantStudio.Factor.FactorOperator as fo

def defFactor() -> List[Factor]:
    """定义因子，返回因子列表"""
    SDB = JYDB().connect()

    FT = SDB.getTable("<财务表名>", args={"CalcType": "最新"})
    RawFactor = FT.getFactor("<基础因子名>")

    FT = SDB.getTable("<行情表名>", args={"LookBack": 0})
    MarketFactor = FT.getFactor("<行情因子名>")

    Derived = rename(RawFactor / MarketFactor, factor_name="<衍生因子英文名>")

    @FactorOperatorized(operator_type="Section", args={"Arity": 1, "DTMode": "多时点"})
    def calcZScore(f, idt, iid, x, args):
        return (x[0] - np.nanmean(x[0], axis=1, keepdims=True)) / np.nanstd(x[0], axis=1, keepdims=True)

    ZScore = calcZScore(Derived, factor_args={"Name": "<标准化因子名>"})

    return [RawFactor, Derived, ZScore]

if __name__ == "__main__":
    SDB = JYDB().connect()
    DTs = SDB.getTradeDay(dt.datetime(2024, 1, 1), dt.datetime(2025, 1, 1))
    IDs = SDB.getStockID(date=dt.datetime(2025, 1, 1))
    factors = defFactor()
    for f in factors:
        print(f"{f.Name}: QSID={f.QSID}")
    data = factors[-1].readData(ids=IDs[:5], dts=DTs[-5:])
    print(data)
```

### 跨 ID 空间聚合因子模式

当衍生因子的输出 ID 空间与描述子（descriptor）不同时（例如从股票级因子聚合到概念板块/行业/指数级），使用 `DescriptorSection` 实现 ID 空间切换。

**核心机制 `DescriptorSection`**: `SectionOperator` 的参数，类型 `List[Optional[List[str]]]`，按描述子顺序指定各描述子使用的 ID：
- `None` → 沿用父因子的 SectionIDs
- `[ids]` → 使用指定 ID 列表

框架在 `forward_compute` 时自动为各描述子传入对应 ID，描述子数据 shape = `(len(idt), len(DescriptorSection[i]))`。

**模板代码**（算子定义在 `defFactor` 外部）：

```python
from typing import List
import numpy as np
from QuantStudio.Factor.Factor import Factor
from QuantStudio.Factor.JYDB import JYDB
from QuantStudio.Factor.BasicOperator import rename
from QuantStudio.Factor.FactorOperation import makeFactorOperator

# ---- 自定义聚合算子 (定义在 defFactor 外部) ----
def _aggregateByGroup(f, idt, iid, x, args):
    """
    f: 因子对象
    idt: 输出时点序列 (len N)
    iid: 输出截面 ID (如概念板块代码, len M)
    x[0]: 成员值因子, shape=(N, len(DescriptorSection[0]))
    x[1]: 分组映射因子, shape=(N, len(DescriptorSection[1]))
    """
    member_values = x[0]
    group_mapping = x[1]
    n_dates, n_groups = len(idt), len(iid)
    group_ints = np.array([int(g) for g in iid])

    result = np.full((n_dates, n_groups), np.nan)
    for ti in range(n_dates):
        group_vals = {}
        for si in range(member_values.shape[1]):
            val = group_mapping[ti, si]
            if not isinstance(val, (list, tuple, np.ndarray)):
                continue
            member_val = member_values[ti, si]
            if np.isnan(member_val):
                continue
            for g in val:
                group_vals.setdefault(g, []).append(member_val)

        for gi, g_int in enumerate(group_ints):
            if g_int in group_vals:
                result[ti, gi] = np.mean(group_vals[g_int])
    return result


def defFactor() -> List[Factor]:
    SDB = JYDB().connect()

    # 基础因子: 股票日收益率
    FT_Quote = SDB.getTable("日行情表", args={"LookBack": 0})
    StockReturn = rename(
        FT_Quote.getFactor("收盘价(元)") / FT_Quote.getFactor("昨收盘(元)") - 1,
        factor_name="stock_daily_return",
    )

    # 基础因子: 分组映射 (M 个成员, 每个值是一个 group id 列表)
    FT_Group = SDB.getTable("<映射表名>", args={
        "MultiMapping": True, "EndDTField": "<结束日期字段>", "EndDTIncluded": False,
    })
    GroupMembership = rename(FT_Group.getFactor("<分组字段>"), factor_name="group_membership")

    all_stocks = SDB.getStockID()

    # 创建算子: DescriptorSection 指定描述子使用全市场股票作为 ID
    GroupAggOperator = makeFactorOperator(
        _aggregateByGroup,
        operator_type="Section",
        args={
            "Arity": 2,
            "DTMode": "多时点",
            "InputFormat": "numpy",
            "DescriptorSection": [all_stocks, all_stocks],
        },
    )

    # 应用算子: 衍生因子的输出 ID 由调用方 readData(ids=...) 动态传入
    GroupFactor = GroupAggOperator(
        StockReturn, GroupMembership,
        factor_args={"Name": "group_daily_return"},
    )

    return [StockReturn, GroupMembership, GroupFactor]
```

**`makeFactorOperator` vs `@FactorOperatorized`**: 当 `DescriptorSection` 的值依赖运行时数据（如 `SDB.getStockID()`），必须用 `makeFactorOperator`（运行时调用）。`@FactorOperatorized` 在 import 时执行，无法获取运行时值。

**关键规则**:
- **不要**在 `factor_args` 中设置 `SectionIDs` — 衍生因子的输出 ID 应由调用方 `readData(ids=...)` 动态传入
- `DescriptorSection` 长度必须等于 `Arity`（描述子数量）
- 算子内部 `iid` = 输出 ID（父因子传入），`x[i]` 的列 = `DescriptorSection[i]` 指定的描述子 ID
- 两个 ID 空间通过映射因子的查找逻辑关联（如概念板块成分映射表）
- 每次 `readData` 会重新读取全市场描述子数据，生产环境建议配合 `FeatherFactorCache` 缓存
