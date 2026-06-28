# 因子开发

因子运算按数据依赖范围分为四类。所有运算由算子（FactorOperator）作用于描述子（依赖因子）产生衍生因子。

## 四类运算

| 运算 | 算子类 | 衍生因子类 | 依赖范围 | 典型场景 |
|------|--------|------------|----------|----------|
| 单点运算 | `PointOperator` | `PointOperation` | 同时点、同证券 | PB、PE |
| 时序运算 | `TimeOperator` | `TimeOperation` | 历史时序、同证券 | MA、EMA |
| 截面运算 | `SectionOperator` | `SectionOperation` | 同时点、全截面 | Z-score |
| 面板运算 | `PanelOperator` | `PanelOperation` | 历史时序 + 全截面 | 双重标准化 |

## 算子核心参数

所有算子继承自 `FactorOperator`，核心参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `OperatorType` | `"Point"/"Time"/"Section"/"Panel"` | 算子类型（冻结） |
| `Arity` | `int/None` | 描述子个数；None 不限制 |
| `DataType` | `"double"/"string"/"object"` | 输出数据类型 |
| `ModelArgs` | `dict` | 传递给 `calculate` 的附加参数 |

## calculate 方法

```python
def calculate(self, f, idt, iid, x: list, args: dict) -> Any:
```

| 参数 | 说明 |
|------|------|
| `f` | 该算子所属的因子对象 |
| `idt` | 当前时点，DTMode="单时点" 时为单个 datetime，"多时点" 时为 list |
| `iid` | 当前 ID，IDMode="单ID" 时为单个 str，"多ID" 时为 list（并发时非全截面） |
| `x` | 描述子当期数据列表，格式取决于算子类型和 DTMode/IDMode |
| `args` | ModelArgs 附加参数 |

## 创建算子的三种方式

### 1. 工厂函数 makeFactorOperator

```python
from QuantStudio.Factor.FactorOperation import makeFactorOperator

def func(f, idt, iid, x, args):
    return x[0] + x[1]

op = makeFactorOperator(func, operator_type="Point",
    args={"Arity": 2, "DTMode": "多时点", "IDMode": "多ID"})
```

### 2. 装饰器 FactorOperatorized（推荐）

```python
from QuantStudio.Factor.FactorOperation import FactorOperatorized

@FactorOperatorized(operator_type="Point",
    args={"Arity": 2, "DTMode": "多时点", "IDMode": "多ID"})
def op(f, idt, iid, x, args):
    return x[0] + x[1]
```

### 3. 子类化（繁琐，尽量避免）

直接继承 `PointOperator`/`TimeOperator`/`SectionOperator`/`PanelOperator` 并实现 `calculate`。

## 各运算类型专属参数

### 单点运算 (PointOperator)

| 参数 | 取值 | 说明 |
|------|------|------|
| `DTMode` | `"单时点"/"多时点"` | 每次处理的时点数 |
| `IDMode` | `"单ID"/"多ID"` | 每次处理的 ID 数 |

选择 `"多时点"+"多ID"` 效率最高。

### 时序运算 (TimeOperator)

| 参数 | 类型 | 说明 |
|------|------|------|
| `LookBack` | `list[int]` | 每个描述子的回溯期数（不含当前时点），长度=Arity |
| `StartDT` | `list[datetime/None]` | 扩张窗口起始时点，None 表示滚动窗口 |
| `iInitFactor` | `int` | 自身迭代因子索引，>=0 表示递归定义（如 EMA），-1 表示无 |

回溯模式：
- **滚动窗口**（StartDT[i]=None）：取最近 LookBack[i]+1 个时点
- **扩张窗口**（StartDT[i]!=None）：从 StartDT[i] 开始的所有历史数据

### 截面运算 (SectionOperator)

| 参数 | 类型 | 说明 |
|------|------|------|
| `DTMode` | `"单时点"/"多时点"` | 每次处理的时点数 |
| `DescriptorSection` | `list[list[str]/None]` | 每个描述子的截面范围 |

截面运算的 `iid` 始终为全体截面 ID，无 IDMode 参数。

### 面板运算 (PanelOperator)

合并了 TimeOperator 和 SectionOperator 的全部参数：`LookBack`、`StartDT`、`iInitFactor`、`DescriptorSection`。

## 运算符重载

Factor 重载了常用 Python 运算符，定义在 `BasicOperator` 中，本质是单点运算：

二元：`+`, `-`, `*`, `/`, `//`, `%`, `**`, `<`, `<=`, `>`, `>=`, `==`, `!=`, `&`, `|`, `^`
一元：`~`, `abs()`, `-`, `+`

```python
from QuantStudio.Factor.BasicOperator import rename
Mid = rename((High + Low) / 2, factor_name="Mid")
```

## 内置算子速查

完整的 30+ 内置算子列表：

**单点运算**：`AsType`(类型转换), `Log`(对数), `Power`(乘方), `NotNull`, `IsIn`, `ApplyArrayFunc`, `Applymap`, `Where`(条件选择), `Fetch`(取子字段), `Sum`, `Max`, `Min`, `Rank`, `Mean`, `Std`, `Regress`(线性回归), `RegressChangeRate`, `ToList`, `ToCompound`

**时序运算**：`Lag`(滞后), `RollingRank`(滚动排名), `RollingMean`(滚动平均), `RollingApply`(通用滚动), `RollingChangeRate`, `RollingRegress`(滚动回归)

**截面运算**：`SectionRank`(截面排名), `Aggregate`(截面聚合), `Disaggregate`(截面反聚合), `ConcatSection`, `ChgSection`, `SectionRegress`(截面回归)

**面板运算**：`PanelRegress`(面板回归)

```python
from QuantStudio.Factor.FactorOperator import Log, RollingRank

LogClose = Log(base=np.e)(Close, factor_args={"Name": "LogClose"})
Rank3 = RollingRank(LookBack=3, ascending=False)(Close)
```
