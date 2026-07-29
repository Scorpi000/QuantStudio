# -*- coding: utf-8 -*-
"""QuantStudio 敏感字段加密/解密工具

使用 Fernet (AES-128-CBC + HMAC-SHA256) 实现认证对称加密。
密钥获取优先级：环境变量 ``QS_SECRET_KEY`` → 文件 ``~/QuantStudioConfig/secret.key`` → 自动生成并持久化。

Usage::

    from QuantStudio.Core._encryption import encrypt_value, decrypt_value

    encrypted = encrypt_value("my_secret_password")
    decrypted = decrypt_value(encrypted)
"""
import os
import base64
import logging
from typing import Optional

from cryptography.fernet import Fernet
from cryptography.fernet import InvalidToken

from QuantStudio import __QS_ConfigPath__

__QS_Logger__ = logging.getLogger("QS")

_ENCRYPT_PREFIX = "ENC:"
_FERNET_INSTANCE: Optional[Fernet] = None
_KEY_FILE_PATH = os.path.join(__QS_ConfigPath__, "secret.key")


def _get_fernet() -> Fernet:
    """获取或初始化 Fernet 实例（进程内单例）。

    密钥来源优先级：
    1. 环境变量 ``QS_SECRET_KEY``
    2. 文件 ``~/QuantStudioConfig/secret.key``
    3. 自动生成并持久化到 ``~/QuantStudioConfig/secret.key``

    Returns:
        Fernet: 已就绪的 Fernet 加密器实例
    """
    global _FERNET_INSTANCE
    if _FERNET_INSTANCE is not None:
        return _FERNET_INSTANCE

    key = _load_key_from_env()
    if key is not None:
        try:
            _FERNET_INSTANCE = Fernet(key)
            return _FERNET_INSTANCE
        except Exception:
            __QS_Logger__.warning("QS_SECRET_KEY 格式无效，尝试其他来源")

    key = _load_key_from_file()
    if key is not None:
        try:
            _FERNET_INSTANCE = Fernet(key)
            return _FERNET_INSTANCE
        except Exception:
            __QS_Logger__.warning("密钥文件格式无效，将自动生成新密钥")

    key = Fernet.generate_key()
    _save_key_to_file(key)
    _FERNET_INSTANCE = Fernet(key)
    __QS_Logger__.info(f"已自动生成加密密钥并保存到: {_KEY_FILE_PATH}")
    return _FERNET_INSTANCE


def _load_key_from_env() -> Optional[bytes]:
    """从环境变量 QS_SECRET_KEY 加载密钥。

    Returns:
        bytes or None: 密钥字节，若环境变量未设置则返回 None
    """
    env_val = os.environ.get("QS_SECRET_KEY")
    if not env_val:
        return None
    key = env_val.strip().encode("utf-8")
    # Fernet 密钥是 32 字节 base64 编码后的字符串，长度 44
    # 但也支持直接传入 32 字节的二进制密钥
    if len(key) == 44 and key.endswith(b"="):
        return key
    # 尝试 base64 解码
    try:
        decoded = base64.urlsafe_b64decode(key + b"=" * (4 - len(key) % 4))
        if len(decoded) == 32:
            return base64.urlsafe_b64encode(decoded)
    except Exception:
        pass
    # 作为原始密码，用 Fernet 方式派生
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    import hashlib
    # Fernet 需要 32 字节 key + 128 位签名 key = 完整 32 字节
    # 直接对 env 值做 SHA256 得到 32 字节
    digest = hashlib.sha256(key).digest()
    return base64.urlsafe_b64encode(digest[:32])


def _load_key_from_file() -> Optional[bytes]:
    """从 ~/QuantStudioConfig/secret.key 文件加载密钥。

    Returns:
        bytes or None: 密钥字节，若文件不存在或内容无效则返回 None
    """
    if not os.path.isfile(_KEY_FILE_PATH):
        return None
    try:
        with open(_KEY_FILE_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return None
            return content.encode("utf-8")
    except Exception:
        return None


def _save_key_to_file(key: bytes) -> None:
    """将密钥保存到 ~/QuantStudioConfig/secret.key 文件。

    Args:
        key: Fernet 密钥字节
    """
    os.makedirs(os.path.dirname(_KEY_FILE_PATH), exist_ok=True)
    # 设置文件权限为仅 owner 可读写（类 Unix），Windows 下仅做 best effort
    try:
        with open(_KEY_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(key.decode("utf-8"))
        # 尝试设置文件权限
        os.chmod(_KEY_FILE_PATH, 0o600)
    except Exception as e:
        __QS_Logger__.warning(f"无法保存密钥文件: {e}")


def encrypt_value(value: str) -> str:
    """加密字符串值。

    Args:
        value: 明文字符串

    Returns:
        以 ``"ENC:"`` 为前缀的 base64 加密字符串
    """
    if not value:
        # 空字符串也加密，保持格式一致
        token = _get_fernet().encrypt(b"")
        return _ENCRYPT_PREFIX + token.decode("utf-8")
    token = _get_fernet().encrypt(value.encode("utf-8"))
    return _ENCRYPT_PREFIX + token.decode("utf-8")


def decrypt_value(encrypted_value: str) -> str:
    """解密字符串值。

    Args:
        encrypted_value: 以 ``"ENC:"`` 为前缀的加密字符串，或普通明文字符串

    Returns:
        解密后的明文字符串

    Raises:
        InvalidToken: 若加密数据被篡改或密钥不匹配
    """
    if not isinstance(encrypted_value, str) or not encrypted_value.startswith(_ENCRYPT_PREFIX):
        return encrypted_value
    token = encrypted_value[len(_ENCRYPT_PREFIX):].encode("utf-8")
    return _get_fernet().decrypt(token).decode("utf-8")


def is_encrypted(value: str) -> bool:
    """检查字符串是否为加密格式。

    Args:
        value: 待检查的字符串

    Returns:
        True 若字符串以 ``"ENC:"`` 开头
    """
    return isinstance(value, str) and value.startswith(_ENCRYPT_PREFIX)
