## Context

当前 `__QS_Args__`（Pydantic v2 BaseModel 子类）是所有 QuantStudio 对象参数集的基类。它有两个不适合作为 Pydantic Field 的"幽灵字段"——`Owner`（所属对象引用）和 `Logger`（日志记录器）。它们是运行时上下文，不是参数，但却作为 Field 存在，导致：(1) 必须启用 `arbitrary_types_allowed=True`；(2) 污染 `model_dump()` 输出；(3) 每个子类的 schema 都继承这些无关字段。

同时，下游项目 QSExt 的 QSGraphDB 通过 `model_dump()` 直接将 `__QS_Args__` 序列化到 Neo4j 图中，敏感字段（如 `Pwd`、`OllamaAPIKey`）以明文存储。

## Goals / Non-Goals

**Goals:**
- 将 `Owner` 和 `Logger` 从 Pydantic Field 迁移到 `PrivateAttr`，去掉 `arbitrary_types_allowed`
- 新增 `secret` Field 元数据标志，标记需要加密的敏感参数
- 新增 `serialize()` / `deserialize()` 方法，对 `secret=True` 字段自动加解密
- 加密方案使用 Fernet (AES-128-CBC + HMAC)，密钥管理支持环境变量/文件/自动生成
- `serialize()` 尊重 `__QS_Object__` 子类的 `model_dump()` 重写

**Non-Goals:**
- 不修改 `model_dump()` 的行为（保持 QSID 计算不变）
- 不自动检测敏感字段名（必须显式 `secret=True`）
- 不修改 `Node.py` 中的 `Context` 类（它有自己的 `arbitrary_types_allowed` 原因）
- 本次不修改 D:\HST\QSExt 项目代码（后续适配）

## Decisions

### 1. Owner/Logger 迁移：PrivateAttr

选择 Pydantic `PrivateAttr` 而非其他方案：
- **PrivateAttr vs 普通实例属性**：PrivateAttr 有 Pydantic 官方支持，定义在类体上清晰可见，可通过 `self._Owner` 访问，不参与任何序列化/验证/schema 生成
- **PrivateAttr vs 保持 Field + 序列化时排除**：从根源解决问题，不需要在 `serialize()` 中硬编码排除列表
- **PrivateAttr vs 完全移除**：保留向后引用能力（`__setattr__` 中需要通知父对象 QSID 失效）

### 2. secret 标志

在 `Field()` 中新增 `secret: bool = False` 参数。`secret=True` 与现有标志完全正交（不自动影响 `exclude`、`repr`、`frozen`）。

```python
Pwd: str = Field(default="", title="密码", secret=True)
```

存储在 Pydantic field metadata 中，通过 `field_info.secret` 或 `model_fields["Pwd"].secret` 访问。

### 3. 序列化格式

```
序列化:
  __QS_Args__.serialize() → {field: value | "ENC:<base64>"}
  __QS_Object__.serialize() → model_dump() 中替换 __qsargs__

加密:
  plaintext → Fernet.encrypt() → base64 encode → "ENC:<base64>"

解密:
  "ENC:<base64>" → base64 decode → Fernet.decrypt() → plaintext
```

选择 `"ENC:"` 前缀标记而非分离式 envelope 结构：自描述、向后兼容、易于人工识别。

### 4. 加密方案：Fernet

Fernet (AES-128-CBC + HMAC-SHA256) 来自 `cryptography` 库：
- 认证加密（ciphertext + MAC），防篡改
- 每条消息独立 IV，相同原文产生不同密文（防重放分析）
- API 极简：`Fernet(key).encrypt(data)` / `Fernet(key).decrypt(token)`
- `cryptography` 新增为核心依赖

### 5. 密钥管理

优先级链：
1. 环境变量 `QS_SECRET_KEY`（若设置）
2. 文件 `~/QuantStudioConfig/secret.key`（若存在）
3. 自动生成 → 持久化到 `~/QuantStudioConfig/secret.key`

每次进程启动时解析一次，缓存 `_FERNET_INSTANCE`。如果三者都不可用（无 env、文件不存在、写入失败），发出警告并使用回退密钥（随机生成，仅本进程有效，反序列化时解密会失败）。

### 6. 子类兼容

`__QS_Object__.serialize()` 调用 `self.model_dump()`（而非直接访问 `_QSArgs`），从而保留 `FactorOperator.model_dump()` 等子类重写中新增的 `__func__` 字段。然后替换 `__qsargs__` 部分为加密版本：

```python
def serialize(self) -> dict:
    result = self.model_dump()
    result["__qsargs__"] = self._QSArgs.serialize()
    return result
```

### 7. FactorUtils.py 迁移

4 个 `__QS_ArgClass__` 子类的 `__init__` 中通过 `data["Owner"]` 访问 Owner。迁移到 `PrivateAttr` 后，Owner 不在构造 `data` 中。将这些 `__init__` 逻辑迁移到 Pydantic 的 `model_post_init(self, context, /)` 方法，其中通过 `self._Owner` 访问：

```python
# 之前
def __init__(self, /, **data: Any) -> None:
    Owner = data["Owner"]
    # ... 使用 Owner._FactorInfo 自动解析字段 ...
    super().__init__(**data)

# 之后
def model_post_init(self, context, /) -> None:
    Owner = self._Owner
    # ... 根据 Owner._FactorInfo 设置 self 上的默认字段值 ...
    super().model_post_init(context)
```

## Risks / Trade-offs

- **[Breaking] Owner/Logger 迁移**：`FactorUtils.py` 中 4 个 `__QS_ArgClass__.__init__` 需要重构为 `model_post_init`。验证方式：运行 `test_JYDB.py`、`test_SQLDB.py`、`test_HDF5DB.py`。
- **[Breaking] 移除 `arbitrary_types_allowed`**：某些 `__QS_Args__` 子类可能依赖此标志来接受非标准字段类型。需全局检查是否有字段类型为 callable 或其他非 JSON 类型。已知 `Context` 类（Node.py）独立维护自己的 `arbitrary_types_allowed`，不受影响。
- **密钥丢失风险**：自动生成密钥存储在 `~/QuantStudioConfig/secret.key`，若文件丢失则已序列化的加密数据无法恢复。建议用户备份密钥文件。
- **安全性非银弹**：Fernet 保护静态数据，但内存中字段仍为明文。不防护运行时内存 dump 攻击。
- **性能开销**：每次 `serialize()`/`deserialize()` 对每个 secret 字段做一次 AES 加密/解密。字段数量通常很少（<10），开销可忽略。

## Open Questions

<!-- 已全部在探索阶段澄清 -->
