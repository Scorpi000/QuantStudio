# BTStorer — 回测结果持久化设计

## 背景

回测模块（BackTest）目前没有任何持久化机制——BTNode/BTReport 的输出 dict 仅存在于内存中，Engine.run() 返回后即丢失。需要实现回测结果的持久化存储，并为未来扩展其他存储后端预留空间。

## 分层架构

参考 Factor 模块的分层设计：

```
FactorDB (抽象基类) → WritableFactorDB (可写接口) → HDF5DB (HDF5 实现)
FactorStorer (Node) 依赖 WritableFactorDB 做实际存储
```

BackTest 采用相同分层：

```
BTResultDB (抽象基类，定义读写接口) → HDF5BTResultDB (目录模式 HDF5 实现)
BTStorer (Node) 依赖 BTResultDB 做实际存储
```

## BTResultDB — 回测结果库抽象基类

定义回测结果的读写接口，所有存储后端的公共契约：

| 方法 | 说明 |
|---|---|
| `writeResult(result, group_name, metadata=None)` | 写入一组回测结果 |
| `readResult(group_name)` → `dict \| None` | 按名称读取一组结果 |
| `listResults(metadata=None)` → `List[str]` | 列出已存储的结果组，支持按 metadata 筛选 |
| `readMetaData(group_name, key=None)` → `Any` | 读取结果组的元信息，key=None 返回全部 |
| `setMetaData(group_name, key=None, value=None, metadata=None)` | 设置结果组的元信息 |

参数说明：
- `result`: 嵌套 dict，叶节点为 DataFrame/Series/str/float 等
- `group_name`: 结果组名称，支持路径层级（如 `"A股/IC/沪深300"`）
- `metadata`: 可选的元信息标签 dict，用于查询筛选（如 `{"资产": "A股", "策略": "IC"}`）

## HDF5BTResultDB — 目录模式 HDF5 实现

每个结果组一个独立 HDF5 文件，目录层级对应 `group_name` 路径层级。

### 参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `Name` | `str` | 结果库名称，默认 `"HDF5BTResultDB"` |
| `MainDir` | `DirectoryPath` | 存放结果文件的根目录路径 |

### 目录结构

```
results_dir/                          ← FilePath
├── A股/                              ← 自动创建的中间目录
│   └── IC/
│       ├── 沪深300.h5                ← group_name="A股/IC/沪深300"
│       └── 中证500.h5                ← group_name="A股/IC/中证500"
├── ETF/
│   └── IC.h5                         ← group_name="ETF/IC"
└── 账户报告.h5                       ← group_name="账户报告"
```

每个 `.h5` 文件内部：数据以 `writeNestedDict2HDF5(result, file, "/", mode="w")` 写入根节点，metadata 存为文件根节点的 HDF5 attrs。

### 设计说明

每个结果组一个独立 HDF5 文件，目录层级对应 `group_name` 路径层级，具有以下优势：

- **并发写入**：不同结果写不同文件，无冲突
- **删除单个结果**：删文件即可
- **文件损坏风险**：各文件独立，互不影响

## BTStorer — 持久化计算图节点

BTStorer 是一个 Node，插入计算图中作为 BTNode/BTReport 的下游：

```
BTNode1 ─┐
BTNode2 ─┼─→ BTReport ─→ BTStorer (存整个报告)
BTNode3 ─┘

IC ─→ BTStorer (单独存 IC 结果)
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `Name` | `str` | `"BTStorer"` | 节点名称 |
| `TargetDB` | `BTResultDB` | 必填 | 目标结果库对象 |
| `GroupName` | `Optional[str]` | `None` | 结果组名称，支持路径层级。None 时按依赖节点 Name 自动生成 |
| `Metadata` | `Optional[dict]` | `None` | 元信息标签，透传给 BTResultDB.writeResult |

### split 模式

参考 FactorStorer，当 BTStorer 有多个依赖时，`__init__` 自动拆分为每个依赖一个子 Storer 实例，解决并行 Engine 的写入冲突：

```
BTNode1(IC) ─┐                          ├─→ BTStorer (GroupName="IC")         → ResultDB
BTNode2(账户)─┼─→ BTStorer(拆分前) ──────┼─→ BTStorer (GroupName="账户报告")   → ResultDB
BTNode3(相关)─┘                          └─→ BTStorer (GroupName="相关性")     → ResultDB
```

## 未来扩展路径

新增存储后端只需：
1. 实现 `BTResultDB` 的子类（如 `FeatherBTResultDB`、`SQLiteBTResultDB`）
2. BTStorer 无需任何修改，通过 `TargetDB` 参数注入不同后端
