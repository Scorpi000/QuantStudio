# 数据源

QuantStudio 提供多种因子库实现，连接不同的数据源。

## 可用因子库

| 因子库 | 模块 | 数据源 | 可写 |
|--------|------|--------|------|
| `HDF5DB` | `QuantStudio.Factor.HDF5DB` | 本地 HDF5 文件 | 是 |
| `JYDB` | `QuantStudio.Factor.JYDB` | 聚源 PostgreSQL 数据库 | 否 |
| `SQLDB` | `QuantStudio.Factor.SQLDB` | 通用 SQL 数据库 | 可选 |
| `BaoStockDB` | `QuantStudio.Factor.BaoStockDB` | BaoStock API | 否 |

## HDF5DB — 本地 HDF5 因子库

可读写的本地因子存储，是衍生因子持久化的主要目标。

```python
from QuantStudio.Factor.HDF5DB import HDF5DB

HDB = HDF5DB(args={"MainDir": "./data/HDF5"}).connect()
print(HDB.TableNames)  # 查看所有表

# 读写
FT = HDB.getTable("stock_cn_day_bar")
data = FT.readData(factor_names=["close", "volume"], ids=IDs, dts=DTs)

HDB.writeData(data=panel, table_name="my_factors", if_exists="update",
              data_type={"factor1": "double", "factor2": "double"})

# 表管理
HDB.renameTable("old", "new")
HDB.deleteTable("my_factors")
HDB.renameFactor("table", "old_factor", "new_factor")
HDB.deleteFactor("table", "factor_name")
```

## JYDB — 聚源数据库

只读，连接聚源 PostgreSQL 数据库，提供股票行情、财务等金融数据。

配置文件：`~/QuantStudioConfig/JYDBConfig.json`

```python
from QuantStudio.Factor.JYDB import JYDB

SDB = JYDB().connect()
print(SDB.TableNames[:5])

# 获取行情因子
FT = SDB.getTable("日行情表", args={"LookBack": 0})
Close = FT.getFactor("收盘价(元)")

# 财务数据支持 CalcType 参数
FT = SDB.getTable("资产负债表_新会计准则", args={"CalcType": "最新"})
Equity = FT.getFactor("归属母公司股东权益合计")
```

表名和字段名可通过 MCP 工具 `search_table_list`、`query_table` 动态查询。

## SQLDB — 通用 SQL 因子库

连接任意 SQL 数据库，支持自定义 SQL 查询定义因子。

## BaoStockDB — BaoStock 在线 API

连接 BaoStock 免费在线 API，主要用于测试和非生产环境。

```python
from QuantStudio.Factor.BaoStockDB import BaoStockDB
BSDB = BaoStockDB().connect()
```

## 因子缓存

`FeatherFactorCache` 和 `FeatherDTCache` 提供基于 Feather 格式的因子数据缓存，可在计算引擎执行时自动缓存中间结果。

```python
from QuantStudio.Factor.FactorCache import FeatherFactorCache, FeatherDTCache

with FeatherFactorCache(args={
    "DTRuler": DTRuler, "PIDs": ["0"],
    "CacheDir": "./cache", "StartMode": "new"
}) as Cache:
    with FactorContext(..., DataCache=Cache) as Context:
        # 引擎执行，中间结果自动缓存
        ...
```
