# 回测框架

## 导入

```python
from QuantStudio.BackTest.api import BTReport
from QuantStudio.BackTest.Strategy import api as Strategy
from QuantStudio.BackTest.SectionFactor import api as SectionFactor
from QuantStudio.BackTest.PerformanceAnalysis import api as PerformanceAnalysis
from QuantStudio.BackTest.Risk import api as BackTestRisk
```

## 策略回测

核心流程：定义策略信号 → 创建账户因子 → 生成报告 → 引擎执行

```python
from QuantStudio.Core.CalcEngine import Engine
from QuantStudio.Core.Node import DTLocalContext, DTInitData
from QuantStudio.Factor.Factor import FactorContext
from QuantStudio.Factor.BasicOperator import rename
from QuantStudio.Factor.FactorCache import FeatherFactorCache
from QuantStudio.BackTest.BackTestModel import BTReport
from QuantStudio.BackTest.Strategy.Strategy import MakeAccount, AccountReport
from QuantStudio.Tools.DateTimeFun import getMonthLastDateTime

# 1. 定义策略信号
FT = HDB.getTable("stock_cn_factor_value")
EP = FT.getFactor("ep_ttm")
FT = HDB.getTable("stock_cn_status")
IfListed = FT.getFactor("if_listed")
Mask = (IfListed == 1)

StrategySignal = fo.SectionRank(ascending=True, uniformization=True)(
    EP, mask=Mask, factor_args={"CalcDTRuler": BalanceDTs}) >= 0.8
StrategySignal = rename(StrategySignal / fo.Aggregate(aggr_func=np.nansum)(StrategySignal),
                        factor_name="Signal")

# 2. 创建账户
FT = HDB.getTable("stock_cn_day_bar")
Price = FT.getFactor("close")
Account = MakeAccount(signal_type="目标权重", init_cash=1e6, start_dt=TestDTs[0])(
    last_price=Price, signal=StrategySignal)

# 3. 基准和报告
FT = HDB.getTable("index_cn_day_bar")
BmkNV = FT.getFactor("close", args={"SectionIDs": ["000905.SH"]})
StrategyReport = AccountReport(account=Account, bmk_nv=BmkNV, args={"GenReport": True})

# 4. 执行
NodeList = [StrategyReport]
Report = BTReport(bt_node_list=NodeList)

with FeatherFactorCache(args={"DTRuler": DTRuler, "CacheDir": "../data/Cache",
                               "StartMode": "new"}) as Cache:
    with FactorContext(DTRuler=DTRuler, SectionIDs=SectionIDs, DataCache=Cache) as Context:
        with Engine() as ExecEngine:
            Rslt = ExecEngine.run([Report], Context,
                fwd_data_list=[DTLocalContext(DTs=TestDTs)],
                init_data_list=[DTInitData(DTRange=(TestDTs[0], TestDTs[-1]))])
```

## 自定义策略

继承 `MakeStrategy` 并实现 `genSignal`:

```python
from QuantStudio.BackTest.Strategy.Strategy import MakeStrategy

class MakeMyStrategy(MakeStrategy):
    def genSignal(self, f, idt, x, last_price, cash, position_num, args):
        # x 是描述子数据; 返回 Series(index=[ID], values=权重)
        iID = x[0].iloc[0].idxmax()
        return pd.Series(1, index=[iID])

Strategy = MakeMyStrategy(signal_type="目标权重", init_cash=1e6, start_dt=TestDTs[0],
    x_lookback=[0], x_section_ids=[None])(factor, last_price=Price)
```

**MakeAccount/MakeStrategy 信号类型**: `"目标权重"`（直接给目标权重），其他类型参考 Strategy 模块。

## 截面因子测试（IC 分析等）

截面因子测试在 `QuantStudio.BackTest.SectionFactor` 中，对因子进行 IC、分位组合、Fama-MacBeth 回归等分析。

## 业绩归因

`QuantStudio.BackTest.PerformanceAnalysis` 提供 Brinson 归因模型。
