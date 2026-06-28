---
name: quantstudio
description: >
  QuantStudio v2.0.0 量化金融研究框架。当用户需要做因子研究、因子开发、
  回测、风险建模、组合优化、或使用 QuantStudio API 时触发。也适用于用户
  询问聚源数据库因子读取、HDF5 因子存储、计算图引擎、Barra 风险模型等场景。
  凡是提到 A 股量化、因子、回测、组合优化、风险模型等关键词时都应使用此 skill。
compatibility: Python 3.12+, QuantStudio v2.0.0
---

# QuantStudio 量化金融研究框架

QuantStudio 是一个面向中国 A 股市场的量化金融研究框架，提供因子研究、回测、风险建模和组合优化等功能，全部构建在自研的计算图引擎之上。

## 核心概念

### QS_Object 与参数系统

所有 QuantStudio 对象（因子库、因子、回测节点、风险库等）均继承自 `__QS_Object__`，封装了一个 `__QS_Args__` 实例和一个日志记录器。

```python
# 构造方式：显式参数 > JSON 配置文件 > 内部默认值
obj = SomeClass(args={"param1": value1}, config_file="path/to/config.json")
```

每个对象有三个基本属性：
- **QSID**：参数的确定性哈希值。QSID 相同的对象行为一致，用于缓存去重
- **Args**：参数集对象（Pydantic v2 BaseModel），支持 `__getitem__` / `__setitem__` / `to_dict()` / `meta()` / `info()`
- **Logger**：日志对象

参数具有 `frozen`（冻结）、`exclude`（影响 QSID）、`repr`（可见性）等标志。配置文件默认从 `~/QuantStudioConfig/` 目录加载。

### Panel 数据对象

Panel 是多维带标签数组（类似于 pandas 早期 Panel），是因子数据和风险数据的标准容器。

```python
from QuantStudio.Core.QSObject import Panel

# items=因子, major_axis=时间, minor_axis=证券代码
p = Panel(data=np.random.rand(2, 4, 3), items=["close", "open"],
          major_axis=DTs, minor_axis=IDs)

# 索引方式
p.loc["close", dt(2025,1,1):dt(2025,1,3), ["000001.SZ", "000003.SZ"]]
p.iloc[0, 0:3, [0, 2]]

# 转换
p.to_frame(filter_observations=False)  # → MultiIndex DataFrame
p.values                                # → numpy.ndarray
```

### 计算图与 Node 生命周期

整个框架建立在计算图引擎之上。每个量化操作都被建模为由 `Engine` 执行的 `Node` 对象组成的有向无环图（DAG）。

每个 Node 有六个生命周期方法，引擎按三个阶段执行：

```
1. init 阶段 (init_compute)：    拓扑遍历，注册节点，填充状态
2. prepare 阶段 (prepare_compute)：IO 操作（数据加载），支持线程并发
3. compute 阶段：                 forward_compute → deps → backward_compute
```

| 方法 | 必须实现 | 说明 |
|------|---------|------|
| `init_compute(path, init_data, context)` | 否 | 初始化阶段 |
| `prepare_compute(prepare_data, context)` | 否 | IO 操作（不应修改全局状态） |
| `compute(path, fwd_data, context)` | 否 | 便捷编排入口（Engine/ParallelEngine 使用） |
| `forward_compute(path, fwd_data, context)` | 否 | 前向传播（父→子） |
| `backward_compute(path, bwd_data_list, context, local_context)` | **是** | 反向传播（子→父），主逻辑 |
| `merge_result(result_list, context)` | 否 | 并行计算后合并结果 |

仅 `backward_compute` 是必须重写的。

### Context 上下文

```python
from QuantStudio.Core.Node import Context, LocalContext, DTLocalContext, DTInitData
```

- **Context**：全局上下文，包含 NodeDict、NodeState、PrepareNodeDict、DataCache、PID 等字段
- **LocalContext**：节点局部上下文，在 forward/backward 间传递
- **DTLocalContext(LocalContext)**：时序类节点的局部上下文，带 DTs 字段
- **DTInitData**：时序类节点的初始化数据，带 DTRange 和 SectionIDs 字段

### 计算引擎

```python
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.ParallelEngine import ParallelEngine
from QuantStudio.Core.TreeEngine import TreeEngine

# 统一入口
result_list = engine.run(node_list, context, init_data_list, fwd_data_list)
```

引擎按 `init → prepare → compute` 三个阶段顺序执行。

## 因子框架快速入门

因子框架是最高频使用的模块，三层数据模型：**FactorDB → FactorTable → Factor**

### 连接因子库并获取因子

```python
# HDF5 本地因子库（可读写）
from QuantStudio.Factor.HDF5DB import HDF5DB
HDB = HDF5DB(args={"MainDir": "./data/HDF5"}).connect()

# 聚源数据库因子库（只读）
from QuantStudio.Factor.JYDB import JYDB
SDB = JYDB().connect()

# 获取因子
FT = HDB.getTable("stock_cn_day_bar")      # 获取因子表
Close = FT.getFactor("close")              # 获取基础因子
High, Low = FT.getFactor("high"), FT.getFactor("low")
```

### 读取因子数据

```python
DTs = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(5)]
IDs = ["000001.SZ", "000002.SZ"]

# 每个因子的数据是一个 DataFrame(index=时间, columns=证券代码)
Close.readData(ids=IDs, dts=DTs)

# 因子表直接 readData 返回 Panel
FT.readData(factor_names=["close", "high"], ids=IDs, dts=DTs)
```

### 定义衍生因子

因子框架支持四种运算类型，由算子（FactorOperator）作用于描述子（依赖因子）产生：

| 运算 | 算子类 | 依赖范围 | 典型场景 |
|------|--------|----------|----------|
| 单点运算 | `PointOperator` | 同时点、同证券 | PB = 总市值 / 股东权益 |
| 时序运算 | `TimeOperator` | 历史时序、同证券 | 移动平均线、EMA |
| 截面运算 | `SectionOperator` | 同时点、全截面 | Z-score 标准化 |
| 面板运算 | `PanelOperator` | 历史时序 + 全截面 | 双重标准化 |

```python
from QuantStudio.Factor.BasicOperator import rename
from QuantStudio.Factor.FactorOperation import makeFactorOperator, FactorOperatorized

# 表达式方式（单点运算，最简便）
Mid = rename((High + Low) / 2, factor_name="Mid")

# 装饰器方式（时序运算）
@FactorOperatorized(operator_type="Time",
    args={"Arity": 1, "DTMode": "多时点", "IDMode": "多ID", "LookBack": [4]})
def calcMA(f, idt, iid, x, args):
    return pd.DataFrame(x[0]).rolling(window=5).mean().values[4:]

MA5 = calcMA(Close, factor_args={"Name": "MA5"})

# 装饰器方式（截面运算）
@FactorOperatorized(operator_type="Section",
    args={"Arity": 1, "DTMode": "多时点"})
def calcZScore(f, idt, iid, x, args):
    return ((x[0].T - np.nanmean(x[0], axis=1)) / np.nanstd(x[0], axis=1)).T

Close_ZScore = calcZScore(Close, factor_args={"Name": "Close_ZScore"})
```

### 用计算引擎执行

```python
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Factor.Factor import FactorContext, FactorLocalContext, FactorInitData

Factors = [Close, Mid, MA5, Close_ZScore]
Context = FactorContext(DTRuler=DTRuler, SectionIDs=SectionIDs)

Rslt = Engine().run(
    Factors, Context,
    fwd_data_list=[FactorLocalContext(DTs=DTs, IDs=IDs)] * len(Factors),
    init_data_list=[FactorInitData(DTRange=(DTs[0], DTs[-1]), SectionIDs=SectionIDs)] * len(Factors)
)
```

### 写入 HDF5 持久化

```python
Data = Panel({Factors[i].Name: Rslt[i] for i in range(len(Factors))})
HDB.writeData(data=Data, table_name="my_factors", if_exists="update",
              data_type={f.Name: f.getMetaData(key="DataType") for f in Factors})
```

## 模块总览

| 模块 | 路径 | 核心功能 |
|------|------|----------|
| Core | `QuantStudio.Core` | 计算图引擎、Node、Context、Panel、缓存 |
| Factor | `QuantStudio.Factor` | 因子库、因子表、因子运算、数据源 |
| BackTest | `QuantStudio.BackTest` | 回测框架、截面因子测试、策略回测、绩效归因 |
| Risk | `QuantStudio.Risk` | 风险数据库、Barra 多因子风险模型 |
| PortfolioConstructor | `QuantStudio.PortfolioConstructor` | 组合优化目标、约束、CVXPY 求解器 |
| Tools | `QuantStudio.Tools` | 数学计算、日期时间、数据预处理、可视化 |

顶层 `QuantStudio.api` 重新导出所有子模块的 API。

## 内置算子速查

`QuantStudio.Factor.FactorOperator` 模块预定义了 30+ 个常用算子：

**单点运算**：`AsType`, `Log`, `NotNull`, `IsIn`, `ApplyArrayFunc`, `Applymap`, `Where`, `Fetch`, `Sum`, `Max`, `Min`, `Rank`, `Mean`, `Std`, `Regress`, `RegressChangeRate`, `ToList`, `ToCompound`

**时序运算**：`Lag`, `RollingRank`, `RollingMean`, `RollingApply`, `RollingChangeRate`, `RollingRegress`

**截面运算**：`SectionRank`, `Aggregate`, `Disaggregate`, `ConcatSection`, `ChgSection`, `SectionRegress`

**面板运算**：`PanelRegress`

## 因子上下文类型

```python
from QuantStudio.Factor.Factor import FactorContext, FactorLocalContext, FactorInitData
```

- **FactorContext(Context)**：新增 `DTRuler`（时点标尺）、`SectionIDs`（默认截面 ID 序列）、`DataCache`
- **FactorLocalContext(DTLocalContext)**：携带当前计算的 `DTs` 和 `IDs`
- **FactorInitData(DTInitData)**：携带 `DTRange`（时点区间）、`SectionIDs`、`SubFactorNames`

## 相关文档

以下参考文件按需读取，每个文件覆盖一个功能领域：

- [因子框架 API](references/factor-framework.md) — FactorDB / FactorTable / Factor / DataFactor 完整 API
- [因子开发](references/factor-development.md) — 四类运算详解、算子创建方式、内置算子速查
- [数据源](references/factor-datasources.md) — JYDB / HDF5DB / SQLDB / BaoStockDB 配置与使用
- [回测框架](references/backtest.md) — BTNode / BTReport、截面因子测试、策略回测、绩效归因
- [风险模型](references/risk-model.md) — RiskDB / HDF5FRDB、BarraModel 多因子风险模型
- [组合优化](references/portfolio-optimization.md) — 优化目标、约束条件、CVXPC 求解器
- [计算图引擎](references/core-engine.md) — Node 生命周期、Context、引擎类型选择、缓存
