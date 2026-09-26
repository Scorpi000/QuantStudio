# 聚源数据库文档 MCP 服务

## 概述

聚源数据库文档 MCP 服务（`jy_doc`）为 AI Agent 提供聚源数据字典平台（dd.gildata.com）的检索与查询能力。通过该服务，Agent 可以：

- 搜索聚源数据库中的数千张表
- 获取表的完整字段说明（字段名、中文名、数据类型、可空性、备注）
- 按数据库分类浏览目录结构
- 查看某张表在数据库中的唯一索引与更新频率

该服务特别适用于生成 SQL 前确认表结构与字段语义的场景，避免因字段名或含义不明确导致的查询错误。

## 架构

```
mcp/jy_doc/                    ← 聚源文档接入包
├── __init__.py
├── models.py                  ← 数据模型（TableDetail, ColumnInfo 等）
├── fetcher.py                 ← Gildata API 抓取器（登录、目录树、表详情）
├── scraper.py                 ← 索引构建（展平目录树为扁平索引）
├── qs_help.py                 ← QuantStudio 用法说明（离线读 JYDBInfo.xlsx）
└── server.py                  ← MCP 服务入口（7 个工具）

scripts/scrape_jy_doc.py       ← 预抓取脚本
tests/test_jy_doc_mcp.py       ← 测试
mcp/.env.example               ← 凭据模板（复制为 mcp/.env 使用）
```

### 关于 `mcp` 这个包名

`mcp/` 与已安装的 MCP SDK 包（`mcp==1.28.1`，`Lib/site-packages/mcp/`）同名，
且仓库根会出现在 `sys.path` 上，因此存在两个陷阱：

1. **`import mcp.jy_doc` 不可用**——不论 `sys.path` 顺序如何，都可能是 SDK 胜出，
   报 `No module named 'mcp.jy_doc'`。**结论：任何代码里都不要出现 `import mcp.*`。**
2. **`python mcp/jy_doc/server.py`（直接按路径运行）时 `__package__` 为空**，
   文件内的 `from .fetcher import ...` 会抛 `attempted relative import with no
   known parent package`。

`server.py` 与 `scripts/scrape_jy_doc.py` 都用同一套办法绕过：**importlib 按文件路径
把 `mcp/jy_doc/` 注册为顶层包 `jy_doc`**（`spec_from_file_location` 配
`submodule_search_locations`），此后 `from jy_doc.fetcher import ...` 与其内部的
相对导入都能正常解析。

> 一个曾经用过但**应避免**的写法是把 `mcp/` 加进 `sys.path` 再 `import jy_doc.server`。
> 那样 `server.py` 会被加载两次——一次作为 `__main__`，一次作为 `jy_doc.server`——
> 于是 `mcp` 实例、`fetcher` 等全局状态各有一份，`isinstance(obj, jy_doc.fetcher.JYDocFetcher)`
> 之类的判断会莫名失效。用 importlib 显式注册成单一模块名可避免这个分叉。

两种入口（`python mcp/jy_doc/server.py` 直接运行、被 MCP 客户端以 stdio 拉起）都已验证可用。

## 数据来源

聚源数据字典平台没有公开 API 文档，数据通过以下流程获取：

1. **登录认证**：`GET /api/captcha` 获取验证码图片与 SESSION cookie，用 `ddddocr` 识别验证码，再 `POST /api/authentication` 提交登录表单
2. **数据库列表**：`GET /api/DDUserPermission` 获取有权限的库（库 ID + 库名）
3. **目录树**：`GET /api/productGroupTreeWithTables/{base_id}/-1/ALL_TREE` 获取某库的完整嵌套目录树
4. **表详情**：分别请求 `/api/table/{id}`、`/api/column/{id}`、`/api/slaveColumn/{id}`、`/api/tableIndexByUnique/{id}`

登录凭据从 `mcp/.env` 的 `JY_DOC_USER` / `JY_DOC_PWD` 读取，或通过命令行参数传入。

### ALL_TREE 的返回结构

建索引只用第 2、3 步（`DDUserPermission` + `ALL_TREE`），不逐表打详情接口。
`ALL_TREE` 返回嵌套节点，**叶子节点自带物理表名**：

```jsonc
// 目录节点
{ "id": 132, "groupName": "上市公司基本资料", "istable": false,
  "groupDesc": "# 库说明 markdown ...", "level": 2, "nodes": [ ... ] }

// 叶子节点（表）
{ "id": 286, "groupName": "公司概况", "istable": true,
  "tableName": "LC_StockArchives",   // ← 物理表名，建索引时取这个
  "description": "LC_StockArchives", // 注意：平台有时把物理表名填进描述栏，不可靠
  "parentId": 132, "parentName": "上市公司基本资料",
  "level": 3, "nodes": [] }
```

因此 `build_flat_index` 无需额外请求即可填出 `base_table_name`。各库目录树单独缓存在
`trees/<base_id>.json`，重建索引可完全离线完成。

> **注意：`/api/captcha` 的 SESSION 是条件性下发的**。实测该接口仅在请求**未携带 SESSION cookie** 时才在响应头返回 `Set-Cookie: SESSION=...`；若请求已带 SESSION，则仍返回 200 与验证码图片，但**不含 `Set-Cookie`**。
>
> 这对 `JYDocFetcher` 有两个直接约束：
>
> - **`_login()` 必须先清空 cookie jar**：`http_session` 是复用的，首次登录后 SESSION 会残留其中；不清理的话第二次登录就取不到 SESSION，抛 `SESSION 获取失败`（重试后表现为 `RetryError`）。
> - **访问必须串行化**：`_login()` 与 `_request()` 共享同一个 `http_session`，并发调用会互相覆盖 SESSION 并来回改写 `self.session_id`，触发重复登录与多余的 401 重试；故 `_request()` 全程持锁执行。

## 数据规模

当前索引覆盖 **11 个数据库、2370 张表**（表数量随平台更新而变化，下表为示例数据）：

| 数据库 | ID | 表数 |
|--------|----|------|
| 聚源新版数据库 | 8 | 1780 |
| 企业数据库 | 19 | 133 |
| 聚源研究数据库 | 38 | 123 |
| 新三板数据库 | 33 | 107 |
| 创新产品数据库 | 13 | 80 |
| 接口数据库 | 9 | 44 |
| 财务数据库 | 31 | 44 |
| 舆情数据库 | 72 | 32 |
| 聚源公司金融信息库 | 35 | 21 |
| AI 数据库 | 20 | 6 |
| 境外债券数据库 | 74 | 0（无权限，自动跳过）|

> 其中约 **448 张**能被 QuantStudio 的 `JYDBInfo.xlsx` 支持，可用 `query_qs_*` 查询用法。

## MCP 工具

| 工具名 | 功能 | 典型用法 |
|--------|------|----------|
| `search_tables` | 多字段加权搜索表 | "搜索上市公司财务相关的表" |
| `get_table_detail` | 获取表的完整字段说明 | "查看公司概况表有哪些字段" |
| `browse_categories` | 浏览数据库分类目录 | "聚源有哪些数据库" |
| `get_database_page` | 获取某库下所有表 | "国内上市公司数据库有哪些表" |
| `search_online` | 实时模糊搜索（兜底） | 本地搜索无结果时使用 |
| `query_qs_read_data_help` | 查某表在 QuantStudio 中读取数据的用法 | "这张表在 QS 里怎么读" |
| `query_qs_get_factor_help` | 查某表在 QuantStudio 中获取因子对象的用法 | "这张表怎么取因子" |

### QuantStudio 用法查询（`query_qs_*`）

这两个工具是 QSAgent 侧 `jy_base_doc` MCP 同名工具的**离线替代**。原实现从
KDB 的 `jy_base_doc` 表读 `qs_info`（由爬虫 + JYDBInfo 比对预先生成），本实现
改为现场读 `QuantStudio/Resource/JYDBInfo.xlsx`，**不连任何数据库**。

入参是 `table_id`（与其余 5 个工具一致），内部链路：

```
table_id → flat_index 查 base_table_name（聚源物理表名）
         → JYDBInfo.TableInfo 按 DBTableName 匹配，得到 QuantStudio 内部表名
         → JYDBInfo.FactorInfo 取该表的因子名（FieldName）与 FieldType
         → 渲染示例代码 + 按 TableClass 给出 args 常用参数
```

**因子名适配**：聚源的中文字段名与 QuantStudio 的因子名并不总是一致，实测
同一批表里约 **26%** 的字段存在差异，且没有单一规则可循：

| 差异形态 | 例子 |
|---|---|
| 加 `_R` 后缀（取值映射后的衍生字段） | 聚源「所属状态」 → QuantStudio「所属状态_R」 |
| 去掉前缀污染 | 聚源「其中:长期健康险责任准备金」 → 「长期健康险责任准备金」 |
| 去「合计」后缀 | 聚源「交易性金融负债合计」 → 「交易性金融负债」 |
| `/` 换成 `-` | 聚源「证券/股证事务代表」 → 「证券-股证事务代表」 |

因此**这两个工具输出的因子名一律取自 JYDBInfo 的 `FieldName`**，可直接用于
`getFactor()` / `readData()`。`get_table_detail` 等其余工具仍返回聚源原始中文名
（它们是"聚源字典"视角），二者对照使用即可定位差异。

映射以**物理字段名（`DBFieldName`）**为键，两侧物理字段名完全一致，所以是
一对一确定的，不依赖中文名做模糊匹配。

**覆盖率**：能否使用取决于 QuantStudio 的 `JYDBInfo.xlsx` 配置（当前 495 张表 /
467 个物理表名），与 jy_doc 索引规模（2370 张表）无关。实测 2370 张里 448 张
（18.9%）可连上——已覆盖 JYDBInfo 侧 467 个物理表名的 **96%**。其余 1922 张表
QuantStudio 本就不支持，工具会返回「QuantStudio 不支持使用该表 X」。

> **静态 args 说明**：原实现用 `FT.Args.info()` 从**活的表对象**取参数说明，
> 本实现无数据库连接，改为按 `TableClass` 查静态表（见 `qs_help._TABLE_TYPE_ARGS`）
> 并附上 JYDBInfo 里配置的 `DefaultArgs`。参数集合与表类型对应关系的权威定义见
> `QuantStudio/Factor/FactorUtils.py` 的各 `SQL_*Table` 基类。

### search_tables 的评分策略

按字段权重加权打分，物理表名精确匹配得分最高：

| 匹配方式 | 加分 |
|----------|------|
| 物理表名完全匹配 | +100 |
| 物理表名包含关键词 | +60 |
| 中文表名包含关键词 | +40 |
| 路径包含关键词 | +20 |
| 描述包含关键词 | +10 |
| 整体关键词命中物理表名 | +30 |
| 整体关键词命中中文表名 | +20 |

支持空格分隔的多关键词（取并集累计加分），以及 `category` 参数限定搜索范围。

> **物理表名来自目录树**：`build_flat_index` 从目录树叶子的 `tableName` 字段
> 取物理表名填入 `base_table_name`。该字段一直存在于 `ALL_TREE` 的返回中，
> 早期版本误写了空字符串占位（注释称"需通过 fetch_table_detail 获取"），
> 导致上面 `+100/+60/+30` 三项永不触发、按物理表名搜索失效。现已在建索引时
> 一次性填好，无需额外请求。

## 快速开始

### 1. 配置凭据

复制 `mcp/.env.example` 为 `mcp/.env`，填入账号信息：

```bash
JY_DOC_USER=your_username
JY_DOC_PWD=your_password
JY_DOC_CACHE=D:\Data\JYDBDoc   # 可选，指定缓存目录
```

需要安装 `ddddocr` 用于验证码识别：

```bash
pip install ddddocr
```

### 2. 构建本地索引

```bash
# 生成到默认目录 D:\Data\JYDBDoc
python scripts/scrape_jy_doc.py

# 指定目录与凭据
python scripts/scrape_jy_doc.py --cache-dir E:/Cache/JYDBDoc --user xxx --pwd xxx
```

该脚本会登录平台、遍历所有可访问数据库的目录树，**逐库增量写入索引文件**（每个库抓完即落盘，中途失败不会丢失已抓取内容）。整个过程约 1 分钟。

### 3. 启动 MCP 服务

```bash
# stdio 模式（默认）
python mcp/jy_doc/server.py

# 指定缓存目录
python mcp/jy_doc/server.py --cache-dir D:/Data/JYDBDoc
```

### 4. 在 Claude Code 中配置

在 `.claude/settings.json` 中添加：

```json
{
  "mcpServers": {
    "jy_doc": {
      "command": "D:/PythonEnv/QS/Scripts/python.exe",
      "args": ["D:/Project/QuantStudio/mcp/jy_doc/server.py", "--cache-dir", "D:/Data/JYDBDoc"],
      "env": {
        "PYTHONPATH": "D:/Project/QuantStudio"
      }
    }
  }
}
```

jy_doc 不依赖 QuantStudio 的 Python 包，只依赖第三方库（fastmcp / requests /
tenacity / pydantic / ddddocr），因此 `PYTHONPATH` 不含 `QuantStudio` 也能运行——
上面写出来是为了与其他 MCP 服务的配置保持一致。

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--cache-dir` | 本地缓存目录 | `D:\Data\JYDBDoc` 或 `JY_DOC_CACHE` |
| `--user` | 平台用户名 | `JY_DOC_USER` 环境变量 |
| `--pwd` | 平台密码 | `JY_DOC_PWD` 环境变量 |
| `--rebuild-cache` | 启动前重建全部缓存 | 关闭 |

## 缓存机制

### 缓存结构

```
<cache_dir>/
├── tree_index.json   ← 搜索索引：flat_index(所有表) + databases + categories(目录树)
├── trees/<base_id>.json   ← 各库的原始目录树（抓取时的中间缓存）
└── tables/<table_id>.json ← 单张表的完整详情
```

### 重要：缓存不会自动更新

**本服务没有任何自动更新机制**，需要特别注意：

- `tree_index.json` 仅在服务**启动时加载一次**，运行期间不会重读。重新抓取索引后必须重启 MCP 服务才能生效。
- `tables/<id>.json` 一旦写入便**永久复用**。`get_table_detail` 固定使用 `use_cache=True`，且缓存判定只检查文件是否存在，没有 TTL、mtime 比对或版本号。

这意味着如果聚源平台更新了表结构（新增字段、修改说明），MCP 会持续返回陈旧数据，且不会报错。

### 手动重建缓存

```bash
# 清除旧缓存并重新抓取
python mcp/jy_doc/server.py --rebuild-cache

# 指定凭据
python mcp/jy_doc/server.py --rebuild-cache --user xxx --pwd xxx
```

该参数会依次删除 `tables/`、`trees/` 和 `tree_index.json`，然后重新抓取，耗时约 1 分钟，期间服务不响应（同步阻塞）。

两点设计说明：

- **同时清除 `trees/`**：这是爬虫自己的目录树缓存。若只清 `tables/` 和索引文件，`fetch_tree` 会命中旧树缓存，重建出的索引仍是陈旧的目录结构，必须一并清除。
- **重建失败不阻塞启动**：若重建因网络或凭据问题失败，服务会记录错误日志并回退到加载现有缓存，不会因此完全无法启动。

## 已知问题

### Windows：工具线程中首次导入原生扩展会死锁

**症状**：首次调用 `get_table_detail`（或其他会触发登录的工具）时，调用**无限挂起**且不返回任何错误；服务端进程仍存活，但直到**客户端断开连接**才继续执行。

**影响范围**：Windows + FastMCP/anyio + stdio 传输。任何在工具函数中**首次** `import` 含 C 扩展的重型模块（numpy、onnxruntime、cv2 等）都会触发。本服务有两处踩中：

1. `ddddocr`（登录时识别验证码）会连带导入 numpy 与 onnxruntime；
2. `query_qs_*` 工具导入 `QuantStudio` / `pandas` 并解析 `JYDBInfo.xlsx`。

第 2 点是本仓库新增工具时实测复现的：服务启动后 `search_tables` 正常返回，紧接着调 `query_qs_get_factor_help` 就卡死，日志停在 `numexpr.utils: NumExpr detected 20 cores`。

**原因**：工具函数运行在 anyio 工作线程中，首次原生扩展导入走的是

```
importlib._bootstrap_external.create_module → _imp.create_dynamic → LoadLibraryExW
```

`LoadLibraryExW` 需要 Windows 的**进程级加载器锁**（loader lock）；而 Windows 在执行任何 DLL 的 `DllMain` —— 包括每次线程创建/退出触发的 `DLL_THREAD_ATTACH` / `DLL_THREAD_DETACH` —— 时**也持有这把锁**。FastMCP 通过 `anyio.to_thread.run_sync` 为工具调用拉起工作线程，Windows 默认的 `ProactorEventLoop` 又用 IOCP + 线程池跑 I/O 回调，两边的线程启停恰好与首次 DLL 加载并发，形成锁顺序环（ABBA）而死锁。

同一段 `import numpy` 在不同位置的实测耗时（均为全新进程）：

| 场景 | 耗时 |
|------|------|
| 主线程 | 0.05s |
| 普通 `ThreadPoolExecutor` 线程 | 0.05s |
| 裸 `anyio.to_thread.run_sync` | 0.05s |
| **运行中的 FastMCP 服务的工作线程** | **死锁** |

前三种情况线程池都是预先建好的，没有并发的线程启停，加载器锁处于空闲状态。

**死锁不会自行解除**：四次实测的客户端超时分别为 60 / 90 / 120 / 600 秒，每次 import 都恰好是在客户端断开后的 0.06 秒内完成（例如 600.00s 超时 → 600.06s 完成）。所以**延长超时或增加重试都无效** —— 重试仍发生在同一个已死锁的线程上。

**处理方式**：在服务启动阶段（主线程、事件循环启动之前）先把重型原生依赖导入一次，见 `mcp/jy_doc/server.py` 的 `_preimport_native_deps()`（由 `init_server()` 调用）。它做两件事：

- `import ddddocr`；
- 调 `qs_help.preload()` 真正解析一次 `JYDBInfo.xlsx`（顺带填充 `lru_cache`，也让首次工具调用不必再付出解析开销）。

此后工具线程里的 `import ddddocr` / `from QuantStudio... import` 只命中 `sys.modules` 缓存，不再触发 DLL 加载，环无从形成。

> **在本仓库新建其他 stdio MCP 服务时同理**：把重型原生依赖放在主线程预导入。凡是工具函数里会 `import pandas / numpy / QuantStudio` 的，都要先跑一遍。

> 在本仓库新建其他 stdio MCP 服务时同理：把重型原生依赖放在主线程预导入。

参考：CPython bpo-33895（`LoadLibraryExW` 持 GIL 调用可致死锁）。

## 测试

### 运行全部测试

```bash
cd C:\Users\hst\Project\QuantStudio

# 默认 mock 模式：88 passed, 14 skipped（真实环境类自动跳过）
python -m pytest tests/test_jy_doc_mcp.py -v

# 真实环境模式：102 passed（需先有 tree_index.json）
python -m pytest tests/test_jy_doc_mcp.py --real -v
```

两种模式对比：

| | mock 模式（默认） | 真实环境模式（`--real`） |
|---|---|---|
| 数据来源 | 内存样例 `SAMPLE_*` 常量 | 真实 `tree_index.json` |
| fetcher | `MagicMock` | 真实 `JYDocFetcher` |
| 外部依赖 | 无 | 缓存文件；详情测试还需网络+凭据 |
| `TestRealEnvironment` | 全部跳过 | 全部执行 |

mock 模式无需网络、数据库或凭据即可随时执行，适合日常开发与 CI。

### 运行单个测试

```bash
# 单个测试函数（用 :: 逐级指定 文件::类::函数）
python -m pytest "tests/test_jy_doc_mcp.py::TestSearchTables::test_chinese_table_name_match" -v

# 整个测试类
python -m pytest "tests/test_jy_doc_mcp.py::TestSearchTables" -v

# 按关键字匹配（-k 支持 and / or / not 表达式）
python -m pytest tests/test_jy_doc_mcp.py -k "markdown or table_detail" -v
python -m pytest tests/test_jy_doc_mcp.py -k "not real" -q

# 真实环境下跑单个测试
python -m pytest "tests/test_jy_doc_mcp.py::TestRealEnvironment::test_real_fetch_table_detail" --real -v
```

注意：真实环境测试即使被单独指定，未传 `--real` 时仍会被跳过（输出 `s`），因为跳过逻辑在 `conftest.py` 的 `pytest_collection_modifyitems` 里按 `real_env` 标记统一处理。

### 常用 pytest 参数

| 参数 | 说明 |
|------|------|
| `-v` | 显示每个测试的名称与结果 |
| `-q` | 精简输出，只显示进度点和汇总 |
| `-s` | 显示 print 输出（默认被捕获） |
| `-x` | 遇首个失败立即停止 |
| `--lf` | 只重跑上次失败的用例 |
| `--ff` | 先跑上次失败的，再跑其余 |
| `-k EXPR` | 按名称筛选测试（and / or / not） |
| `-m MARK` | 按标记筛选，如 `-m "real_env"` |
| `--collect-only` | 只列出会被收集的测试，不执行 |
| `-p no:cacheprovider` | 不写 `.pytest_cache` |

测试专用参数（定义于 `tests/conftest.py`）：

| 参数 | 说明 |
|------|------|
| `--real` | 启用真实环境模式，运行 `TestRealEnvironment` 下的测试 |
| `--jy-doc-cache DIR` | 指定真实缓存目录，默认取 `JY_DOC_CACHE`，再回退到 `D:\Data\JYDBDoc` |

```bash
# 示例：真实环境 + 指定缓存目录
python -m pytest tests/test_jy_doc_mcp.py --real --jy-doc-cache E:/Cache/JYDBDoc -v
```

### 测试类一览

| 测试类 | 覆盖内容 |
|--------|----------|
| `TestLoadIndex` | 索引文件加载与异常处理 |
| `TestSearchTables` | `search_tables` 加权评分与过滤 |
| `TestGetTableDetail` | `get_table_detail` 文本/Markdown 格式化 |
| `TestBrowseCategories` | `browse_categories` 分类目录 |
| `TestGetDatabasePage` | `get_database_page` 单库表列表 |
| `TestCollectTables` | `_collect_tables` 目录树递归展平 |
| `TestSearchOnline` | `search_online` 在线搜索 |
| `TestServerInit` | 默认缓存目录与 `init_server` |
| `TestRebuildCache` | `rebuild_cache` 与 `init_server(rebuild=True)` |
| `TestBuildFlatIndex` | `scraper.build_flat_index` 展平与物理表名填充 |
| `TestQsHelp` | `qs_help` 离线生成 QuantStudio 用法说明 |
| `TestQsHelpTools` | 两个 `query_qs_*` 工具的 table_id 解析 |
| `TestToolRegistration` | MCP 工具注册完整性 |
| `TestNativePreload` | 主线程原生依赖预热 |
| `TestRealEnvironment` | 真实索引集成测试（需 `--real`） |

### 关于 test_real_fetch_table_detail

该测试通过 Gildata API 拉取某张表的完整字段信息（真实网络请求）。若目标表已有本地缓存（`<cache_dir>/tables/<id>.json`），则直接读缓存，不会联网。缺少凭据且无缓存时会自动跳过。

## 与其他 MCP 服务的对比

| 维度 | tinysoft_doc | akshare_doc | jy_doc |
|------|--------------|-------------|--------|
| 文档站 | 天软 TSDN | Sphinx | 聚源自建平台 |
| 认证 | 无 | 无 | 需要（含验证码识别）|
| 数据来源 | HTML 解析 | API + HTML | JSON API |
| 缓存更新 | 手动 | 手动 | 手动（`--rebuild-cache`）|
| 特有依赖 | — | akshare | ddddocr、QuantStudio |
