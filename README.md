QuantStudio 基本使用说明和约定参见: [通则和约定](./docs/通则和约定.ipynb), 建议在使用下面具体模块前先浏览此文档。


# 因子框架

* 因子框架参见: [因子框架](./docs/因子框架/基本框架.ipynb)
* 因子使用快速入门参见: [QuickStart](./docs/因子框架/QuickStart.ipynb)

## 因子库

QuantStudio 内置的因子库主要有：
* 聚源因子库(JYDB)，外部因子库，仅支持读取数据，使用参见: [JYDB](./docs/因子框架/JYDB.ipynb)
* 基于 HDF5 文件的本地因子库(HDF5DB)，支持读取和写入数据，使用参见: [HDF5DB](./docs/因子框架/HDF5DB.ipynb)
* 基于 BaoStock 的外部 API 因子库(BaoStockDB)，仅作测试用，不推荐生产中使用，说明参见: [HDF5DB](./docs/因子框架/BaoStockDB.ipynb)

## 因子开发

参见: [因子开发](./docs/因子框架/因子开发.ipynb)


# 回测框架

回测框架参见: [回测框架](./docs/回测框架/基本框架.ipynb)

## 截面因子回测

参见: [截面因子回测](./docs/回测框架/截面因子测试.ipynb)

## 策略回测

参见: [策略回测](./docs/回测框架/策略回测.ipynb)

## 业绩归因

参见: [业绩归因](./docs/回测框架/业绩归因.ipynb)


# 风险模型

风险模型参见: [风险模型](./docs/风险模型/基本框架.ipynb)

## 数据读写

参见: [风险数据读写](./docs/风险模型/数据读写.ipynb)


# 组合优化

组合优化基本框架参见: [组合优化](./docs/组合优化/基本框架.ipynb)

组合优化的示例参见: 
* [均值方差模型](./docs/组合优化/均值方差模型.ipynb)
* [风险预算模型](./docs/组合优化/风险预算模型.ipynb)


# 其他

QuantStudio 底层的计算图框架参见: [计算图框架](./docs/Core/计算图框架.ipynb)