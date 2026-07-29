## Why

`__QS_Args__` 中存在两个架构问题：(1) `Owner` 和 `Logger` 作为 Pydantic Field 存在，但它们本质上是运行时上下文引用而非参数，导致 `model_dump()` 污染、`arbitrary_types_allowed=True` 等副作用；(2) 缺少安全的序列化/反序列化机制，QSGraphDB 等下游项目将包含密码（`Pwd`、`OllamaAPIKey` 等）的 QSArgs 明文存入 Neo4j，存在安全隐患。

## What Changes

- **BREAKING**: 将 `Owner` 和 `Logger` 从 Pydantic Field 迁移到 `PrivateAttr`，不再参与 `model_dump()`、验证和 schema
- **BREAKING**: 移除 `__QS_Args__` 的 `arbitrary_types_allowed=True`，加强类型校验
- 新增 `secret` Field 标志，标记需要加密的敏感参数
- 新增 `__QS_Args__.serialize()` / `__QS_Args__.deserialize()` 方法，基于 Fernet (AES-128-CBC + HMAC) 对 `secret=True` 字段自动加密/解密
- 新增 `__QS_Object__.serialize()` / `__QS_Object__.deserialize()` 类方法，尊重子类 `model_dump()` 重写
- `FactorUtils.py` 中 4 个依赖 `Owner` 作为构造参数的 `__QS_ArgClass__.__init__` 迁移为 `model_post_init` 模式
- 新增 `cryptography` 为核心依赖

## Capabilities

### New Capabilities

- `qsargs-secret-serialization`: `__QS_Args__` 和 `__QS_Object__` 的序列化/反序列化，支持 `secret=True` 字段的 Fernet 自动加解密，密钥来源：环境变量 `QS_SECRET_KEY` → `~/QuantStudioConfig/secret.key` → 自动生成
- `qsargs-private-attrs`: `Owner` 和 `Logger` 从 Pydantic Field 迁移到 `PrivateAttr`，对应清理 `__QS_Args__` 和 `FactorUtils.py` 中所有相关引用

### Modified Capabilities

<!-- No existing capabilities to modify. -->

## Impact

- **Core**: `QuantStudio/Core/__init__.py` — `__QS_Args__` 基类定义、`__QS_Object__.__init__` 构造逻辑
- **Factor**: `QuantStudio/Factor/FactorUtils.py` — 4 个 `__QS_ArgClass__` 子类的 `__init__` 迁移到 `model_post_init`
- **依赖**: 新增 `cryptography` 为核心依赖 (`requirements.txt`)
- **下游**: `D:\HST\QSExt\QSRegistry` — `_serialization.py`、`QSGraphDB.py` 需在后续适配（将 `model_dump()` 调用替换为 `serialize()`），但本次变更不阻塞下游
