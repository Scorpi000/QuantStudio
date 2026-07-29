## 1. Owner/Logger 迁移到 PrivateAttr

- [x] 1.1 将 `__QS_Args__` 中 `Owner` 和 `Logger` 从 Pydantic Field 改为 `PrivateAttr`，移除 `model_config` 中的 `arbitrary_types_allowed=True`
- [x] 1.2 更新 `__QS_Args__` 内部对 `self.Owner` → `self._Owner`、`self.Logger` → `self._Logger` 的 3 处引用
- [x] 1.3 更新 `__QS_Object__.__init__` 中的注入方式：从 dict 注入改为构造后设置 `self._QSArgs._Owner = self` 和 `self._QSArgs._Logger = self._QS_Logger`
- [x] 1.4 迁移 `FactorUtils.py` 中 4 个 `__QS_ArgClass__.__init__` 到显式参数模式（`_owner`、`_logger`）

## 2. 新增 secret 标志

- [x] 2.1 在 `Core/__init__.py` 中新增 `QSField()` 包装器，通过 `json_schema_extra` 存储 `secret` 标志
- [x] 2.2 为 `QSSQLObject.__QS_ArgClass__` 的 `Pwd` 字段添加 `secret=True`
- [ ] 2.3 为 `QSGraphDB.__QS_ArgClass__` 的 `OllamaAPIKey` 字段添加 `secret=True`（在 QSExt 项目中，需单独处理）

## 3. 加密基础设施

- [x] 3.1 在 `QuantStudio/Core/` 下新增 `_encryption.py` 模块，提供 `_get_fernet()` 密钥管理、`encrypt_value()`、`decrypt_value()` 函数
- [x] 3.2 将 `cryptography` 添加到 `requirements.txt`

## 4. serialize / deserialize 方法

- [x] 4.1 在 `__QS_Args__` 上实现 `serialize()` 方法：遍历 `model_fields`，对 `secret=True` 字段加密后输出 `"ENC:<base64>"`
- [x] 4.2 在 `__QS_Args__` 上实现 `deserialize(data)` 类方法：识别 `"ENC:"` 前缀并解密，重建实例
- [x] 4.3 在 `__QS_Object__` 上实现 `serialize()` 方法：调用 `self.model_dump()`，替换 `__qsargs__` 为加密版本
- [x] 4.4 在 `__QS_Object__` 上实现 `deserialize(data)` 类方法：通过 `__class__` 定位目标类，解密 args 后调用 `cls(args=...)` 重建
- [x] 4.5 `QSSQLObject.__getstate__` / `__setstate__` 已正确处理 Connection 序列化（bool 标记 + 重建时重连），无需额外修改

## 5. 测试验证

- [x] 5.1 运行现有测试 `test_JYDB.py`（1 通过）、`test_SQLDB.py`（15 通过）、`test_HDF5DB.py`（14 通过）确认无回归
- [x] 5.2 运行 `test_Core_Engine.py`（35 通过）确认核心计算图无回归
- [x] 5.3 手动验证 `serialize()` / `deserialize()` 端到端流程：加密、解密、QSID 一致性全部通过
