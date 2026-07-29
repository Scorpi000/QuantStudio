## ADDED Requirements

### Requirement: Owner 和 Logger 使用 PrivateAttr

`__QS_Args__` 中的 `Owner` 和 `Logger` SHALL 在底层定义为 Pydantic `PrivateAttr`（`_Owner` / `_Logger`），同时提供 `@property` 只读访问。它们 MUST 不参与 `model_dump()` 输出、不参与 Pydantic 验证、不出现在 `model_fields` 中。

#### Scenario: Owner 不作为 Field 暴露

- **WHEN** 任意 `__QS_Args__` 实例调用 `model_dump()`
- **THEN** 返回的字典中不包含 `Owner` 键

#### Scenario: Logger 不作为 Field 暴露

- **WHEN** 任意 `__QS_Args__` 实例调用 `model_dump()`
- **THEN** 返回的字典中不包含 `Logger` 键

#### Scenario: 通过 property 访问 Owner 和 Logger

- **WHEN** `__QS_Object__.__init__` 构造后将 `self` 赋值给 `self._QSArgs._Owner`
- **THEN** `self._QSArgs.Owner` 返回所属的 `__QS_Object__` 实例
- **AND** `self._QSArgs.Logger` 返回关联的日志记录器

### Requirement: 移除 arbitrary_types_allowed

`__QS_Args__` 的 `model_config` SHALL 不再包含 `arbitrary_types_allowed=True`。`extra` MUST 保持为 `'forbid'`。需要 `arbitrary_types_allowed` 的子类（如 `Node.__QS_ArgClass__`）MUST 在自己类上显式声明。

#### Scenario: 字段类型严格校验

- **WHEN** 创建 `__QS_Args__` 子类实例时传入类型不匹配的字段值
- **THEN** Pydantic 抛出 `ValidationError`

### Requirement: __setattr__ 失效父对象 QSID

当 `__QS_Args__` 的字段值被修改时，系统 SHALL 通过 `self.Owner._QS_ID = None` 失效所属对象的 QSID 缓存（如果 Owner 不为 None）。

#### Scenario: 字段变更触发父对象 QSID 失效

- **WHEN** 修改一个关联了 `__QS_Object__` 的 `__QS_Args__` 实例的任意 Field 值
- **THEN** 该 `__QS_Object__` 实例的 `_QS_ID` 被设置为 `None`

### Requirement: FactorUtils 中 Args 子类使用显式参数

`FactorUtils.py` 中所有在 `__init__` 中需要访问 `Owner` 的 `__QS_ArgClass__` 子类 SHALL 通过显式 `_owner`/`_logger` 参数接收，而非从 `data` 字典中获取。这些参数 MUST 在调用 `super().__init__()` 之前消费掉，不会透传给 Pydantic。

#### Scenario: JYDB 因子表 Args 构造

- **WHEN** JYDB 创建因子表 Args 实例
- **THEN** `_owner` 参数携带 JYDB 实例引用
- **AND** 构造 `**data` 中不包含 `Owner` 键

#### Scenario: 现有测试通过

- **WHEN** 运行 `test_JYDB.py`、`test_SQLDB.py`、`test_HDF5DB.py`
- **THEN** 所有测试用例 MUST 通过
