# 计算图引擎

QuantStudio 的计算图引擎是框架底层基础设施，所有模块（因子、回测、风险）都构建在其上。

## Node 生命周期方法

```python
from QuantStudio.Core.Node import Node, Context, LocalContext, DTLocalContext, DTInitData
```

六个方法，只有 `backward_compute` 必须重写：

| 方法 | 签名 | 说明 |
|------|------|------|
| `init_compute` | `(path, init_data, context) -> List[Any]` | 拓扑初始化，可修改 context 全局变量 |
| `prepare_compute` | `(prepare_data, context)` | IO 操作，不可修改全局状态，支持线程并发 |
| `compute` | `(path, fwd_data, context) -> Any` | 便捷编排入口，内部递归 forward+backward |
| `forward_compute` | `(path, fwd_data, context) -> (List[Any], Any)` | 前向传播，返回(下游数据, 局部上下文) |
| `backward_compute` | `(path, bwd_data_list, context, local_context) -> Any` | **必须重写**，反向传播主逻辑 |
| `merge_result` | `(result_list, context) -> Any` | 合并并行计算结果 |

## Context 全局上下文

```python
context = Context()
```

核心字段：
- `Mode`: `"PRD"/"DEBUG"` — 运行模式
- `NodeDict`: `{QSID: Node}` — 所有已注册节点
- `NodeState`: `{QSID: Any}` — 节点在 init_compute 中维护的临时状态
- `PrepareNodeDict`: `{准备ID: (QSID, 准备数据)}` — init 填充 → prepare 消费
- `DataCache`: `Optional[Cache]` — 数据缓存
- `ExtraData`: `dict` — 扩展数据

并行相关：
- `PID`, `PIDList` — 进程标识
- `SplitType`: `"连续切分"/"间隔切分"` — 数据切分方式
- `Event`, `Sub2MainQueue` — 多进程同步
- `TaskExecutor`, `MaxWorkers` — 线程池

方法：
- `split(n)` — 将 context 切分为 n 份用于并行
- `getUpdateData()` / `updateContext()` — 子进程同步状态
- 支持 `with Context() as ctx:` 上下文管理器

## 计算引擎类型

```python
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.ParallelEngine import ParallelEngine
from QuantStudio.Core.TreeEngine import TreeEngine
```

所有引擎统一入口：
```python
result_list = engine.run(node_list, context, init_data_list=None, fwd_data_list=None)
```

### Engine — 顺序引擎

基础顺序执行，调用 `Node.compute()` 作为编排入口（内部递归完成 forward→deps→backward）。

### ParallelEngine — 多进程引擎

将 context 拆分到多个进程中执行，适合大数据量并行计算。

### TreeEngine — 树形引擎

树形结构引擎变体，直接编排 `forward_compute` 和 `backward_compute`，实现更灵活的遍历和节点级并发调度。

## 缓存

```python
from QuantStudio.Core.Cache import FileCache
```

`FileCache` 提供基于 HDF5 的节点输出缓存，避免重复计算。在 `FactorContext` 中通过 `DataCache` 字段传递。

因子框架中常用的是 `FeatherFactorCache`（基于 Feather 格式），在回测等场景中用于加速：
```python
with FeatherFactorCache(args={"DTRuler": DTRuler, "PIDs": ["0"],
    "CacheDir": "./cache", "StartMode": "new"}) as Cache:
    with FactorContext(..., DataCache=Cache) as Context:
        ...
```

## 计算图可视化

```python
from QuantStudio.Tools.Visualization import node2dict, dict2mermaid

NodeDict = node2dict(node_list)          # Node → 嵌套字典
MermaidStr = dict2mermaid(NodeDict, direction="TD")  # → Mermaid 流程图
```

## 最小节点示例

```python
class MyNode(Node):
    def __init__(self, deps=[], args={}, config_file=None, **kwargs):
        super().__init__(deps=deps, args=args, config_file=config_file, **kwargs)

    def backward_compute(self, path, bwd_data_list, context, local_context=None):
        # bwd_data_list 来自依赖节点的计算结果
        return process(bwd_data_list)
```
