## ADDED Requirements

### Requirement: secret Field 标志

Pydantic `Field()` 的 `json_schema_extra` SHALL 支持 `{"secret": True}` 标志，标记需要加密的敏感参数。序列化代码可通过 `field_info.json_schema_extra.get("secret", False)` 检测。

#### Scenario: 声明 secret 字段

- **WHEN** 在 `__QS_Args__` 子类中定义 `Pwd: str = Field(default="", json_schema_extra={"secret": True})`
- **THEN** 序列化时该字段会被自动加密

#### Scenario: secret 默认为 False

- **WHEN** 在 `__QS_Args__` 子类中定义 `Name: str = Field(default="test")`
- **THEN** 序列化时该字段原样输出

### Requirement: secret 与现有标志正交

`secret` 标志 SHALL NOT 自动修改 `exclude`、`repr`、`frozen` 等现有标志的值。这几个标志 MUST 保持独立配置。

#### Scenario: secret 不影响 exclude

- **WHEN** 定义 `Pwd: str = Field(default="", json_schema_extra={"secret": True})`（不设置 exclude）
- **THEN** `Args.model_fields["Pwd"].exclude` 返回 `False`（默认值）

### Requirement: __QS_Args__.serialize 方法

`__QS_Args__` SHALL 提供 `serialize()` 方法，返回 `dict`。对于非 secret 字段 MUST 输出原值；对于 `secret=True` 的字段 MUST 输出 `"ENC:<base64>"` 格式的 Fernet 加密字符串。

#### Scenario: 普通字段原样输出

- **WHEN** 调用 `args.serialize()` 且 args 包含 `Name: str = "JYDB"`（非 secret）
- **THEN** 返回的 dict 中 `"Name"` 的值为 `"JYDB"`

#### Scenario: secret 字段加密输出

- **WHEN** 调用 `args.serialize()` 且 args 包含 `Pwd: str = "secret123"`（secret=True）
- **THEN** 返回的 dict 中 `"Pwd"` 的值以 `"ENC:"` 开头
- **AND** 值为 Fernet 加密后 base64 编码的字符串

### Requirement: __QS_Args__.deserialize 方法

`__QS_Args__` SHALL 提供 `deserialize(data: dict)` 类方法。对于以 `"ENC:"` 开头的值 MUST 自动解密；对于其他值 MUST 原样保留。返回重建的 `__QS_Args__` 实例。兼容旧格式中含 `Owner`/`Logger` 键的数据（自动跳过）。

#### Scenario: 加密字段解密恢复

- **WHEN** 调用 `MyArgs.deserialize({"Name": "JYDB", "Pwd": "ENC:<encrypted_base64>"})`
- **THEN** 返回的 `MyArgs` 实例中 `Pwd` 为原始明文

#### Scenario: 普通字段原样恢复

- **WHEN** 调用 `MyArgs.deserialize({"Name": "JYDB"})`
- **THEN** 返回的 `MyArgs` 实例中 `Name` 为 `"JYDB"`

### Requirement: __QS_Object__.serialize 方法

`__QS_Object__` SHALL 提供 `serialize()` 方法。MUST 调用 `self.model_dump()` 获取基础结构（以保留子类重写，如 `FactorOperator` 的 `__func__`），然后替换 `__qsargs__` 部分为 `self._QSArgs.serialize()` 的结果。

#### Scenario: 序列化包含加密后的 args

- **WHEN** 对包含敏感字段的 `__QS_Object__` 实例调用 `serialize()`
- **THEN** 返回的 dict 中 `__qsargs__` 内的敏感字段值为 `"ENC:"` 开头
- **AND** dict 保留 `__type__` 和 `__class__` 字段

#### Scenario: 子类扩展字段保留

- **WHEN** 对 `FactorOperator` 实例（其 `model_dump()` 包含 `__func__`）调用 `serialize()`
- **THEN** 返回的 dict 中 SHALL 包含 `__func__` 字段

### Requirement: __QS_Object__.deserialize 类方法

`__QS_Object__` SHALL 提供 `deserialize(data: dict)` 类方法。MUST 解密 `__qsargs__` 中的敏感字段，调用 `cls(args=decrypted_args)` 重建实例。

#### Scenario: 从序列化数据重建对象

- **WHEN** 调用 `MyObj.deserialize({"__type__": "__QS_Object__", "__class__": "MyObj", "__qsargs__": {...}})` 
- **THEN** 返回一个 `MyObj` 实例
- **AND** 其 Args 中敏感字段已解密为明文

### Requirement: Fernet 密钥管理

加密密钥 SHALL 按以下优先级获取：(1) 环境变量 `QS_SECRET_KEY`；(2) 文件 `~/QuantStudioConfig/secret.key`；(3) 自动生成并持久化到 `~/QuantStudioConfig/secret.key`。密钥在每个进程中 MUST 只解析一次并缓存。

#### Scenario: 从环境变量获取密钥

- **WHEN** 环境变量 `QS_SECRET_KEY` 已设置
- **THEN** 使用该值作为 Fernet 密钥

#### Scenario: 从文件获取密钥

- **WHEN** 环境变量 `QS_SECRET_KEY` 未设置但 `~/QuantStudioConfig/secret.key` 存在
- **THEN** 读取文件内容作为 Fernet 密钥

#### Scenario: 自动生成并持久化密钥

- **WHEN** 环境变量和密钥文件均不存在
- **THEN** 自动生成一个 Fernet 密钥
- **AND** 持久化到 `~/QuantStudioConfig/secret.key`

### Requirement: cryptography 为核心依赖

`cryptography` 包 SHALL 添加到 `requirements.txt` 中。

#### Scenario: 安装核心依赖

- **WHEN** 运行 `pip install -r requirements.txt`
- **THEN** `cryptography` 被安装
