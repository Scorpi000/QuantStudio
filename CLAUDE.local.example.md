# 约定

* 生成文档、注释、skill等时使用中文

# 运行环境

Python：使用 conda 的 QS312 环境，位置是：D:\miniforge\envs\QS312

# 数据库

## 聚源数据库

* 数据库类型：postgresql
* 数据库：JYDB
* 数据库的连接信息可以在文件 "~/QuantStudioConfig/JYDBConfig.json" 中
* 数据库内容：股票、基金等证券的基本信息、行情、财务等金融数据
* 表的说明信息使用相关工具检索

## 图数据库

* 数据库类型：neo4j
* 数据库：qs-neo4j
* 数据库的连接信息可以在文件 "~/QuantStudioConfig/Neo4jDBConfig.json" 中
* 数据库内容：用于存储因子、算子等信息