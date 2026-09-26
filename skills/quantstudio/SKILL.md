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

本文件提供核心概念与常用 API 速查。**详细的使用方法和完整示例都在 `references/` 下**——按需读取，不要凭猜测写 API 调用。

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

参数具有 `frozen`（冻结）、`exclude`（影响 QSID）、`repr`（可见性）等标志。配置文件默认从 `~/QuantStudioConfig/` 目录加载。细节见 [通则和约定](references/通则和约定.md)。

### Panel 数据对象

Panel 是多维带标签数组（类似于 pandas 早期 Panel），是因子数据和风险数据的标准容器。

```python
from QuantStudio.Core.QSObject import Panel

# items=因子, major_axis=时间, minor_axis=证券代码
p = Panel(data=np.random.rand(2, 4, 3), items=["close", "open"],
          major_axis=DTs, minor_axis=IDs)

p.loc["close", dt(2025,1,1):dt(2025,1,3), ["000001.SZ", "000003.SZ"]]  # 标签索引
p.iloc[0, 0:3, [0, 2]]                                                 # 位置索引
p.to_frame(filter_observations=False)                                  # → MultiIndex DataFrame
p.values                                                               # → numpy.ndarray
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

仅 `backward_compute` 是必须重写的。完整示例见 [计算图框架](references/Core/计算图框架.md)。

### 引擎

```python
from QuantStudio.Core.CalcEngine import Engine, StackEngine
from QuantStudio.Core.ParallelEngine import ParallelEngine
from QuantStudio.Core.TreeEngine import TreeEngine

result_list = engine.run(node_list, context, init_data_list, fwd_data_list)
```

引擎按 `init → prepare → compute` 三个阶段顺序执行。选型与差异见 [计算引擎](references/Core/计算引擎.md)。

### Context 上下文

- **Context**：全局上下文，包含 NodeDict、NodeState、PrepareNodeDict、DataCache、PID 等字段
- **LocalContext**：节点局部上下文，在 forward/backward 间传递
- **DTLocalContext(LocalContext)**：时序类节点的局部上下文，带 DTs 字段
- **DTInitData**：时序类节点的初始化数据，带 DTRange 和 SectionIDs 字段

```python
from QuantStudio.Core.Node import Context, LocalContext, DTLocalContext, DTInitData
```

## 因子框架

因子框架是最高频使用的模块，三层数据模型：**FactorDB（因子库）→ FactorTable（因子表）→ Factor（因子）**。完整架构见 [因子框架/基本框架](references/因子框架/基本框架.md)，端到端示例见 [QuickStart](references/因子框架/QuickStart.md)。

### 四类运算

衍生因子由算子（`FactorOperator`）作用于描述子（依赖因子）产生：

| 运算 | 算子类 | 装饰器 `operator_type` | 依赖范围 | 典型场景 |
|------|--------|----------------------|----------|----------|
| 单点运算 | `PointOperator` | `"Point"` | 同时点、同证券 | PB = 总市值 / 股东权益 |
| 时序运算 | `TimeOperator` | `"Time"` | 历史时序、同证券 | 移动平均线、EMA |
| 截面运算 | `SectionOperator` | `"Section"` | 同时点、全截面 | Z-score 标准化 |
| 面板运算 | `PanelOperator` | `"Panel"` | 历史时序 + 全截面 | 双重标准化 |

要点：
- 每类运算由 `DTMode`（`"多时点"` / `"单时点"`）和 `IDMode`（`"多ID"` / `"单ID"`）决定调用粒度。**一次处理越多数据效率越高**，优先用 `DTMode="多时点"`。
- 时序运算的窗口行为由 `LookBack`、`StartDT`、`iInitFactor` 三个参数组合决定：滚动窗口 / 扩张窗口 / 是否自身迭代（EMA 类需自身迭代，且系统会强制关闭缓存）。
- 运算符重载（`+ - * /`）等价于单点运算，用 `rename` 命名结果。

创建算子的两种方式（创建方式与全部参数见 [因子开发](references/因子框架/因子开发.md)）：

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
```

### 内置算子速查

`QuantStudio.Factor.FactorOperator` 模块预定义了 30+ 个常用算子：

**单点运算**：`AsType`, `Log`, `NotNull`, `IsIn`, `ApplyArrayFunc`, `Applymap`, `Where`, `Fetch`, `Sum`, `Max`, `Min`, `Rank`, `Mean`, `Std`, `Regress`, `RegressChangeRate`, `ToList`, `ToCompound`

**时序运算**：`Lag`, `RollingRank`, `RollingMean`, `RollingApply`, `RollingChangeRate`, `RollingRegress`

**截面运算**：`SectionRank`, `Aggregate`, `Disaggregate`, `ConcatSection`, `ChgSection`, `SectionRegress`

**面板运算**：`PanelRegress`

用 `qs_help(算子)` 查看参数，用法示例见 [因子开发](references/因子框架/因子开发.md)。

### 执行与落盘

```python
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Factor.Factor import FactorContext, FactorLocalContext

Context = FactorContext(DTRuler=DTRuler, SectionIDs=SectionIDs)
Rslt = Engine().run(
    Factors, Context,
    fwd_data_list=[FactorLocalContext(DTs=DTs, IDs=IDs, SectionIDs=SectionIDs)] * len(Factors)
)

# 写入 HDF5 持久化
Data = Panel({Factors[i].Name: Rslt[i] for i in range(len(Factors))})
HDB.writeData(data=Data, table_name="my_factors", if_exists="update",
              data_type={f.Name: f.getMetaData(key="DataType") for f in Factors})
```

- **FactorContext(Context)**：新增 `DTRuler`（时点标尺）、`SectionIDs`（默认截面 ID 序列）、`DataCache`
- **FactorLocalContext(DTLocalContext)**：携带当前计算的 `DTs`、`IDs` 和 `SectionIDs`

### 因子库连接

```python
from QuantStudio.Factor.HDF5DB import HDF5DB   # 本地 HDF5（可读写）
from QuantStudio.Factor.JYDB import JYDB       # 聚源数据库（只读）
from QuantStudio.Factor.SQLDB import SQLDB     # 通用 SQL 数据库
from QuantStudio.Factor.BaoStockDB import BaoStockDB  # BaoStock（仅测试用）

HDB = HDF5DB(args={"MainDir": "./data/HDF5"}).connect()
SDB = JYDB().connect()
DTs = [dt.datetime(2025, 1, 1) + dt.timedelta(i) for i in range(5)]
IDs = ["000001.SZ", "000002.SZ"]

Close = HDB.getTable("stock_cn_day_bar").getFactor("close")   # 单因子
Data = FT.readData(factor_names=["close", "high"], ids=IDs, dts=DTs)  # 因子表 → Panel
```

各因子库的表类型（WideTable / NarrowTable / FeatureTable / MappingTable / ConstituentTable / FinancialTable 等）与配置方式见对应文档：[HDF5DB](references/因子框架/HDF5DB.md)、[JYDB](references/因子框架/JYDB.md)、[SQLDB](references/因子框架/SQLDB.md)、[BaoStockDB](references/因子框架/BaoStockDB.md)。

## 参考文档索引

按使用频率排序。**动手写 QuantStudio 代码前，先读对应文档**——里面是经过验证的完整示例。

### 因子框架（最高频）

- [基本框架](references/因子框架/基本框架.md) — 三层数据模型、基础因子/衍生因子概念、与计算图的关系
- [QuickStart](references/因子框架/QuickStart.md) — 端到端示例：连接数据 → 获取因子 → 衍生计算 → 读取结果
- [因子开发](references/因子框架/因子开发.md) — 四类运算详解、算子创建方式、DTMode/IDMode/LookBack 组合
- [HDF5DB](references/因子框架/HDF5DB.md) / [JYDB](references/因子框架/JYDB.md) / [SQLDB](references/因子框架/SQLDB.md) / [BaoStockDB](references/因子框架/BaoStockDB.md) — 各因子库的配置与表类型

### Core 计算图

- [通则和约定](references/通则和约定.md) — QS_Object 参数系统、QSID、Panel 完整用法
- [计算图框架](references/Core/计算图框架.md) — Node 生命周期、Context、自定义节点
- [计算引擎](references/Core/计算引擎.md) — Engine / StackEngine / ParallelEngine / TreeEngine 选型
- [缓存](references/Core/缓存.md) — Cache / DTCache / FileDTCache / FeatherDTCache / FactorCache

### 回测框架

- [基本框架](references/回测框架/基本框架.md) — BTNode / BTReport、执行流程、算子-节点分离设计
- [截面因子测试](references/回测框架/截面因子测试.md) — IC 分析、分位组合、相关性、收益分解
- [策略回测](references/回测框架/策略回测.md) — MakeAccount / MakeStrategy / AccountStats / AccountReport
- [业绩归因](references/回测框架/业绩归因.md) — Brinson 归因模型
- [风险模型测试](references/回测框架/风险模型测试.md) — 回测偏差检验

### 风险模型

- [基本框架](references/风险模型/基本框架.md) — 风险库三层结构、与因子框架的对应关系
- [数据读写](references/风险模型/数据读写.md) — HDF5RDB / HDF5FRDB 的读、写、元信息管理
- [风险模型](references/风险模型/风险模型.md) — Barra 多因子风险模型、可配置因子

### 组合优化

- [基本框架](references/组合优化/基本框架.md) — 目标函数与约束条件总览、CVXPC 求解器
- [均值方差模型](references/组合优化/均值方差模型.md) — MeanVarianceObjective 及各类约束
- [风险预算模型](references/组合优化/风险预算模型.md) — RiskBudgetObjective 风险平价
- [组合优化策略](references/组合优化/组合优化策略.md) — 将优化器接入策略回测

## 相关 Skill

- `jydb-add-table` — 向聚源数据库配置（`JYDBInfo.xlsx`）添加新表，或排查已配置表的字段类型、ID 映射、表类型问题
