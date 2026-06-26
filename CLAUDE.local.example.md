# 约定

* 生成文档、注释、skill等时使用中文
* Python 文档字符串使用 Google 风格
* 测试脚本放到项目的 tests 目录下，功能性脚本放到项目的 scripts 目录下
* 编写功能脚本（通常位于项目的 scripts 目录下）时，一定要将脚本的实现逻辑和使用方法以模块文档字符串的形式写到脚本的开始位置


# 运行环境

Python：使用 QS312 环境，位置是：D:\PythonEnv\QS312\Scripts\python.exe

# 数据库

## 聚源数据库

* 数据库类型：postgresql
* 数据库：JYDB
* 数据库的连接信息可以在文件 "~/QuantStudioConfig/JYDBConfig.json" 中
* 数据库内容：股票、基金等证券的基本信息、行情、财务等金融数据
* 表的说明信息使用相关工具检索