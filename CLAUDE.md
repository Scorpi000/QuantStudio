# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此仓库中工作时提供指导。

## 项目概述

QuantStudio v2.0.0 是一个面向中国 A 股市场的量化金融研究框架（Python 3.12+，GPLv3）。提供因子研究、回测、风险建模和组合优化等功能，全部构建在自研的计算图引擎之上。

## 构建与测试命令

```bash
# 安装核心依赖
pip install -r requirements.txt

# 以可编辑模式安装
pip install -e .

# 可选依赖（SQL 数据库、scikit-learn、ipykernel）
pip install -r requirements_optional.txt

# 运行单个测试文件
python tests/test_Core_Engine.py

# 运行全部测试（基于 unittest，无需 pytest）
python -m unittest discover -s tests -p "test_*.py"
```

测试使用 `unittest` 框架，采用 `if __name__ == "__main__": unittest.main()` 模式——每个测试文件也可以直接执行。项目中无 pytest、tox 或 CI 配置。

## 架构

### 计算图基础（`QuantStudio/Core/`）

整个框架建立在计算图引擎之上。每个量化操作（因子计算、回测、优化）都被建模为由 `Engine` 执行的 `Node` 对象组成的有向无环图（DAG）。

- **`__QS_Args__`** — Pydantic v2 `BaseModel` 子类。系统中所有参数均为带类型校验的 Pydantic 字段。支持 JSON 配置文件加载、QSID 生成（参数的确定性哈希）以及元信息自省。参数具有 `frozen`（冻结）、`exclude`（从 QSID 中排除）和 `repr`（可见性）标志。
- **`__QS_Object__`** — 所有领域对象的基类。封装一个 `__QS_Args__` 实例和一个日志记录器。通过 `__init__(args={}, config_file=None)` 构造，优先级为：显式参数 > JSON 配置文件 > 内部默认值。配置文件默认从 `~/QuantStudioConfig/` 加载。
- **`Node`** — 计算图节点。具有 `Deps`（依赖节点）和生命周期方法：`init_compute` → `prepare_compute` → `compute`。节点构成 DAG；引擎按拓扑顺序初始化后执行它们。
- **`Context`** — 贯穿计算图的全局执行上下文。包含节点注册表、节点状态、多进程相关的进程/PID 信息、缓存引用以及线程池执行器。
- **`LocalContext`** — 每个节点的本地上下文，可拆分以支持并行执行。
- **`Engine`** — 顺序计算引擎。按 `init` → `prepare`（可选 I/O 并发）→ `compute` 顺序执行。
- **`ParallelEngine`** — 多进程引擎，将上下文拆分到多个进程中执行。
- **`TreeEngine`** — 树形结构引擎变体。
- **`Panel`** — 多维带标签数组（核心数据结构，类似于带标签的张量）。在 numpy/pandas 之上封装了带标签的轴。
- **`Cache`** — 抽象缓存；`FileCache` 为节点输出提供基于 HDF5 的缓存。

### 因子框架（`QuantStudio/Factor/`）

最大的模块。因子是产生时序-截面数据的 `Node` 子类。

- **`DataFactor`** — 持有字面量数据（numpy 数组或 pandas DataFrame）的叶子节点。
- **`Factor`** — 计算因子的抽象基类。
- **`FactorOperator`** — 封装计算内核，带类型化参数（`OperatorType`：Point/Time/Section/Panel）。算子可序列化（dill），定义了元数、数据类型和复合输出类型。
  - `PointOperation` — 跨描述符的逐元素操作。
  - `SectionOperation` — 每个时间点的截面操作。
  - `TimeOperation` — 每个资产的时序操作。
  - `PanelOperation` — 全面板操作。
  - `makeFactorOperator` — 装饰器/工厂，从函数创建 `FactorOperator`。
- **`JYDB`** — 聚源外部因子数据库连接器（只读，1517 行）。
- **`HDF5DB`** — 本地基于 HDF5 的因子数据库（读写）。
- **`BaoStockDB`** — BaoStock API 连接器（仅用于测试，非生产环境）。
- **`FeatherDTCache` / `FeatherFactorCache`** — 基于 Feather 格式的因子数据缓存。
- **`FactorStorer`** — 已计算因子的持久化层。

### 回测框架（`QuantStudio/BackTest/`）

- **`BTReport`** — 回测报告容器。
- **`SectionFactor/`** — 截面因子测试：IC 分析、分位组合、相关性、收益分解（Fama-MacBeth 回归）。
- **`Strategy/`** — 策略回测：`MakeStrategy`、`MakeAccount`、`AccountReport`、配置策略。
- **`PerformanceAnalysis/`** — Brinson 归因模型。
- **`Risk/`** — 回测偏差检验。

### 风险模型（`QuantStudio/Risk/`）

- **`HDF5RDB`** / **`HDF5FRDB`** — 基于 HDF5 的风险数据库（读写和只读两种变体）。
- **`RiskModel/`** — Barra 多因子风险模型实现，支持可配置因子。

### 组合优化（`QuantStudio/PortfolioConstructor/`）

- **`BasePC`** — 目标函数（`MeanVarianceObjective`、`MaxDiversificationObjective`、`RiskBudgetObjective`）和约束条件（`BudgetConstraint`、`WeightConstraint`、`FactorExposeConstraint`、`VolatilityConstraint`、`TurnoverConstraint`、`NonZeroNumConstraint`）。
- **`CVXPC`** — 基于 cvxpy 的凸优化求解器。由于 cvxpy 较重，导入放在 try/except 中。

### 工具模块（`QuantStudio/Tools/`）

提供数学计算、日期时间处理、数据类型转换、数据预处理、文件 IO、SQL 数据库抽象、现金流计算、绩效分析、风险度量、交易函数、策略测试（1123 行）、多进程执行以及基于 matplotlib 的可视化等工具。

### 包 API 接口

各子模块通过 `api.py` 对外暴露公共 API：
- `QuantStudio.Core.api` → `Node`、`Context`、`LocalContext`、`DTLocalContext`、`DTInitData`、`Engine`、`ParallelEngine`、`TreeEngine`、`Panel`
- `QuantStudio.Factor.api` → `DataFactor`、`HDF5DB`、`JYDB`、`BaoStockDB`、操作类型、`rename`、`fo`（因子算子）、缓存类
- `QuantStudio.BackTest.api` → `BTReport`、`SectionFactor.*`、`Strategy.*`、`PerformanceAnalysis.*`、`Risk.*`
- `QuantStudio.Risk.api` → `HDF5FRDB`、`HDF5RDB`
- `QuantStudio.PortfolioConstructor.api` → 目标函数、约束条件、`CVXPC`
- `QuantStudio.Tools.api` → `Math`、`DateTime`、`Strategy`、`Preprocess`、`File`、`Visualization`、`genAvailableName`

顶层 `QuantStudio.api` 重新导出所有子模块的 API。

## 核心模式

- **QSID**：每个 `__QS_Object__` 和 `__QS_Args__` 都有一个 `QSID` 属性——其参数的确定性哈希值。QSID 相同的对象产生相同的结果，从而实现缓存和去重。
- **参数优先级**：`显式参数 > JSON 配置文件 > Pydantic 字段默认值`。配置文件若未给定绝对路径，则从 `~/QuantStudioConfig/` 中查找。
- **Node 生命周期**：`init_compute(path, init_data, context)` — 递归拓扑初始化 → `prepare_compute(prepare_data, context)` — 数据加载 → `compute(path, fwd_data, context)` — 实际计算。节点通过 `init_compute` 的返回值请求其依赖节点的初始化数据。
- **算子**：计算逻辑封装在 `FactorOperator` 对象中（而非裸函数）。使用 `makeFactorOperator` 创建算子。算子声明其类型（Point/Section/Time/Panel）、元数、数据类型以及是否产生复合输出。
- **配置**：`~/QuantStudioConfig/` 中的 JSON 配置文件存储外部数据库（JYDB、SQL 数据库）的连接参数。切勿硬编码凭据——使用配置文件。
- **日志**：使用 `__QS_Logger__`（模块级日志记录器，名称为 `'QS'`）或每个对象的 `self.Logger`。通过 `setDefaultLogLevel(logging.INFO)` 设置日志级别。

## 文档

`docs/` 目录下的 16 个 Jupyter Notebook 涵盖了完整的 API 接口（均为中文）。建议从 `docs/通则和约定.ipynb` 开始了解约定规范。Notebook 按模块组织：Core（计算图）、Factor、BackTest、Risk、Portfolio。
