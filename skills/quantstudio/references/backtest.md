# 回测框架

回测框架建立在计算图引擎之上，核心由 BTNode 和 BTReport 组成，按功能分为四个子模块。

## 整体架构

```
BTReport（报告容器）
  └── BTNode（回测计算节点）× N
        └── 因子节点（数据源）
```

回测遵循 Node 生命周期，由 Engine 驱动执行。因子节点和回测节点通过 Deps 串联成完整 DAG，统一执行。

## 执行流程（三层嵌套）

```python
with FeatherFactorCache(...) as Cache:       # 最外层：缓存
    with FactorContext(...) as Context:       # 中间层：上下文
        with Engine() as ExecEngine:          # 最内层：引擎
            Output, = ExecEngine.run([Report], Context,
                fwd_data_list=[DTLocalContext(DTs=TestDTs)],
                init_data_list=[DTInitData(DTRange=(TestDTs[0], TestDTs[-1]))])
```

## BTNode — 回测计算节点

所有回测分析类的基类。

```python
class BTNode(Node):
    # 参数：Name, GenReport(是否自动生成 HTML 报告)
    def genReport(self, output: dict) -> str  # 子类必须重写，返回 HTML 字符串
```

### 生命周期

1. `init_compute`：合并依赖节点的 DTRange，记录到 context.NodeState
2. `forward_compute`：将时点列表传递给子节点
3. `backward_compute`：收集子节点结果，执行分析逻辑；若 GenReport=True 则调用 genReport

## BTReport — 报告容器

```python
class BTReport(Node):
    def __init__(self, bt_node_list: List[BTNode], args={}, ...)

    @staticmethod
    def genOutputReport(output_list, name_list=None) -> str  # 手动合并多个报告
```

`backward_compute` 自动遍历所有子节点，调用各自的 `genReport()`，用分隔线拼接成完整 HTML。

## 四个子模块

### SectionFactor — 截面因子测试

测试因子选股能力的工具集：

| 类 | 功能 |
|----|------|
| `IC` | IC 分析（RankIC/NormIC），移动平均和统计 |
| `ICDecay` | IC 衰减分析 |
| `MultiPortfolio` | 分位数组合分析 |
| `FactorTurnover` | 因子换手率分析 |
| `SectionCorrelation` | 因子截面相关性 |
| `FamaMacBethRegression` | Fama-MacBeth 回归（收益分解） |

```python
from QuantStudio.BackTest.SectionFactor.IC import CalcIC, IC

# CalcIC 是算子，IC 是回测节点（算子-节点分离模式）
FactorIC = CalcIC(descriptor_ids=SectionIDs, lookback=31,
    period_lookback=1, corr_method="spearman")(
    *FactorList, price=Price, mask=Mask,
    factor_args={"CalcDTRuler": BalanceDTs})

ICNode = IC(FactorIC, args={"Name": "IC测试", "RollingAvgPeriod": 2, "GenReport": True})
```

### Strategy — 策略回测

```python
from QuantStudio.BackTest.Strategy import api as Strategy

# MakeStrategy：构建交易策略信号
# MakeAccount：创建账户，计算净值与收益率
# AccountReport：生成策略回测报告
```

### PerformanceAnalysis — 绩效归因

```python
from QuantStudio.BackTest.PerformanceAnalysis import api as PerformanceAnalysis

# BrinsonModel：Brinson 归因模型，分解超额收益来源
#   - 配置效应（Allocation Effect）
#   - 选股效应（Selection Effect）
#   - 交互效应（Interaction Effect）
```

### Risk — 风险模型测试

```python
from QuantStudio.BackTest.Risk import api as Risk

# BiasTest：回测偏差检验，验证风险预测的准确性
```

## 自定义回测节点

```python
class MyBackTest(BTNode):
    class __QS_ArgClass__(BTNode.__QS_ArgClass__):
        Name: str = Field(default="我的回测", frozen=True)

    def __init__(self, factor, args={}, config_file=None, **kwargs):
        super().__init__(deps=[factor], args=args, config_file=config_file, **kwargs)

    def backward_compute(self, path, bwd_data_list, context, local_context=None):
        data = bwd_data_list[0]
        output = {"结果": data}
        if self._QSArgs.GenReport:
            output["Report"] = self.genReport(output)
        return output

    def genReport(self, output):
        html = "<h3>我的回测报告</h3>"
        html += output["结果"].to_html()
        return html
```
