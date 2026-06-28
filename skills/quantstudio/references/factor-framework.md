# 因子框架 API

因子框架采用三层数据模型：**FactorDB（因子库）→ FactorTable（因子表）→ Factor（因子）**。
每个因子继承自计算图节点（Node），多个因子的依赖关系构成 DAG。

## FactorDB — 因子库

```python
class FactorDB(__QS_Object__):
    def connect(self) -> Self           # 连接数据源
    def disconnect(self) -> int         # 断开连接
    def TableNames(self) -> List[str]   # 表名列表
    def getTable(table_name, args)      # 获取因子表
```

`WritableFactorDB(FactorDB)` 增加：`writeData`、`renameTable`、`deleteTable`、`renameFactor`、`deleteFactor`、`setTableMetaData`、`setFactorMetaData`。

## FactorTable — 因子表

```python
class FactorTable(__QS_Object__):
    def FactorDB(self) -> FactorDB
    def FactorNames(self) -> List[str]           # 因子名称列表
    def getFactor(factor_name, args) -> Factor   # 获取因子对象
    def getID(ifactor_name, idt) -> List[str]    # 获取 ID 序列
    def getDateTime(ifactor_name, start_dt, end_dt) -> List[datetime]  # 获取时点序列
    def readData(factor_names, ids, dts) -> Panel # 读取因子表数据
    def getMetaData(key)                          # 获取元信息
```

`FactorTable` 实现了 `__getitem__`，支持：
```python
FT["factor_name"]                  # 等价于 getFactor("factor_name")
FT[factor_names, dts, ids]         # 等价于 readData(...)
```

## Factor — 因子（核心）

```python
class Factor(Node):
    def FactorTable(self) -> FactorTable         # 所属因子表
    def Descriptors(self) -> List[Factor]         # 依赖的描述子列表
    def getID(idt) -> List[str]                   # 获取 ID 序列
    def getDateTime(iid, start_dt, end_dt) -> List[datetime]  # 获取时点序列
    def readData(ids, dts, **kwargs) -> DataFrame # 读取因子数据
    def getMetaData(key)                          # 获取元信息
    def new(args, **kwargs) -> Factor             # 创建同类型新因子
```

`readData` 返回 `DataFrame(index=时间, columns=证券代码)`。

### readData 的额外参数
- `dt_ruler`：时点标尺序列，对时序/面板运算影响数据回溯范围
- `section_ids`：截面 ID 序列，对截面/面板运算影响全体截面范围

## DataFactor — 数据因子

直接赋予字面量数据的因子，主要用于测试。

```python
from QuantStudio.Factor.Factor import DataFactor

# 标量：广播到所有 ID 和时点
F = DataFactor(data=1)

# DataFrame：只有 DataFrame 中有值的时点/ID 才返回数据
F = DataFactor(data=pd.DataFrame(np.random.randn(2, 2), index=DTs, columns=IDs))

F.readData(ids=IDs, dts=DTs)
```

## FactorStorer — 因子存储节点

将计算后的衍生因子写入可写因子库持久化保存。

```python
class FactorStorer(Node):
    # 参数：TargetFDB（目标因子库）、TargetTable（目标表名）、
    #       IfExists（"update"/"replace"/"append"）
    # 依赖节点：待写入的因子列表
```

## FactorContext 上下文类型

```python
from QuantStudio.Factor.Factor import FactorContext, FactorLocalContext, FactorInitData
```

| 类型 | 基类 | 关键字段 | 用途 |
|------|------|----------|------|
| `FactorContext` | `Context` | `DTRuler`, `SectionIDs`, `DataCache` | 全局上下文 |
| `FactorLocalContext` | `DTLocalContext` | `DTs`, `IDs` | 局部上下文 |
| `FactorInitData` | `DTInitData` | `DTRange`, `SectionIDs`, `SubFactorNames` | 初始化数据 |
