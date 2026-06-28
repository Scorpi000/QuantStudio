# 风险模型

风险数据模型分两类：无结构风险模型（协方差矩阵）和多因子风险模型（Barra 风格）。

## 数据模型

风险数据也是三层结构：**RiskDB（风险库）→ RiskTable（风险表）→ 时点数据**

每个风险表的数据为 Panel(items=时点, major_axis=证券代码, minor_axis=证券代码)。

## RiskDB — 风险库

```python
from QuantStudio.Risk.HDF5RDB import HDF5RDB, HDF5FRDB

class RiskDB(__QS_Object__):
    def connect(self) -> Self
    def TableNames(self) -> List[str]
    def getTable(table_name, args) -> RiskTable
    def writeData(table_name, idt, icov, **kwargs)
    def renameTable(old, new)
    def deleteTable(table_name)
```

### HDF5RDB — 无结构风险库

存储直接的协方差矩阵，基于 HDF5 本地文件。

### HDF5FRDB — 多因子风险库

继承自 `RiskDB`，支持多因子风险数据的存取。`writeData` 可接受五个可选参数：
```python
HDF5FRDB.writeData(table_name, idt,
    cov=None,              # 证券协方差阵 V
    factor_data=None,      # 因子暴露矩阵 X (Panel: items=因子, major=dts, minor=ids)
    factor_cov=None,       # 因子协方差阵 F (Panel: items=dts, major=因子, minor=因子)
    specific_risk=None,    # 特异性风险 Δ (DataFrame: index=dts, columns=ids)
    factor_ret=None,       # 因子收益率
    specific_ret=None)     # 特异性收益率
```

## FactorRT — 多因子风险表

```python
class FactorRT(RiskTable):
    def FactorNames(self) -> List[str]
    def readFactorCov(dts) -> Panel           # 因子协方差阵
    def readFactorData(dts, ids) -> Panel     # 因子暴露
    def readSpecificRisk(dts, ids) -> DataFrame  # 特异性风险
    def readFactorReturn(dts) -> DataFrame    # 因子收益率
    def readSpecificReturn(dts, ids) -> DataFrame  # 特异性收益率
    def readCov(dts, ids) -> Panel            # 证券协方差阵（自动合成）
    def readData(data_item, dts)              # 通用数据读取
```

## BarraModel — Barra 多因子风险模型

参考 Barra CNE5 方法论的多因子风险模型实现。

```python
from QuantStudio.Risk.RiskModel.BarraModel import BarraModel

model = BarraModel(
    name="MyModel",
    factor_table=factor_table,     # 因子表（提供因子暴露数据）
    risk_db=HDF5FRDB(...),         # 目标风险库
    table_name="barra_risk",
    config_file=None
)

# 配置
model.setRegressDateTime(dts)      # 设置截面回归时点
model.setRiskESTDateTime(dts)      # 设置风险估计时点

# 执行
model.run()                        # 生成风险数据
```

### 方法论核心步骤

1. **因子收益率估计**：截面加权回归（EUE3 方法论）
2. **因子协方差矩阵估计**：EWMA + Newey-West 自相关修正 + Eigenfactor Risk Adjustment + Volatility Regime Adjustment（CHE2 方法论）
3. **特异性风险估计**：EWMA + 结构化模型 + Bayesian Shrinkage + Volatility Regime Adjustment（EUE3 方法论）

### 多因子风险分解

$$\mathbf{V} = \mathbf{X} \cdot \mathbf{F} \cdot \mathbf{X}^T + \mathbf{\Delta}$$

- X：因子暴露矩阵（N×K）
- F：因子收益率协方差矩阵（K×K）
- Δ：特异性风险对角矩阵（N×N）
