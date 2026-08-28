# -*- coding: utf-8 -*-
import os
import html
import json
import logging
from typing import Any, Optional, Literal, Union, Dict, List

import numpy as np
import pandas as pd
from pydantic_core import PydanticUndefinedType
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from pydantic.fields import FieldInfo

from QuantStudio import __QS_ConfigPath__
from QuantStudio.Tools.DataTypeConversionFun import dict2html, dict2markdown, formatValue2MD
from QuantStudio.Tools.DataTypeFun import dict2id
from QuantStudio.Core._encryption import encrypt_value, decrypt_value, is_encrypted
from QuantStudio.Core.MPLogger import logger as _MPLogger, init_logger as _initMPLogger, shutdown as _shutdownMPLogger


def _is_secret_field(field_info: FieldInfo) -> bool:
    """检查 Pydantic Field 是否标记为 secret。"""
    extra = getattr(field_info, "json_schema_extra", None) or {}
    return bool(extra.get("secret", False))


def setDefaultLogLevel(level=logging.INFO):
    """设置默认日志级别（使用多进程日志系统）

    Args:
        level: 日志级别，默认 INFO
    """
    global __QS_Logger__
    _initMPLogger(level=level)
    __QS_Logger__ = _MPLogger

def setDefaultLogger(logger):
    """设置默认日志器（兼容旧接口，但不推荐使用）

    Args:
        logger: 日志器实例
    """
    global __QS_Logger__
    __QS_Logger__ = logger

# 使用多进程日志系统
__QS_Logger__ = _MPLogger
_initMPLogger(level=logging.INFO)

class __QS_Error__(Exception):
    """Quant Studio 系统错误"""
    pass


class __QS_Args__(BaseModel):
    """QuantStudio 参数对象"""

    _Owner: Any = PrivateAttr(default=None)
    _Logger: logging.Logger = PrivateAttr(default=None)
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)

    @property
    def Owner(self) -> Any:
        """所属的 ``__QS_Object__`` 实例（只读）。"""
        return self._Owner

    @property
    def Logger(self) -> logging.Logger:
        """日志记录器（只读）。"""
        return self._Logger

    def __init__(self, /, _owner: Any = None, _logger: logging.Logger = None, **data: Any) -> None:
        """重写 __init__ 是为了让子类的 ``model_post_init`` 中能通过
        ``self._Owner`` / ``self._Logger`` 访问所属对象和日志器。

        ``_owner`` / ``_logger`` 必须在这里拦截，不能透传到
        ``BaseModel.__init__``，否则 ``extra='forbid'`` 会拒绝它们。

        PrivateAttr 在 ``super().__init__()`` 之前设置，确保子类的
        ``model_post_init`` 中即可访问 ``self._Owner``。
        """
        object.__setattr__(self, "_Owner", _owner)
        object.__setattr__(self, "_Logger", _logger or __QS_Logger__)
        super().__init__(**data)
        object.__setattr__(self, "_Owner", _owner)
        object.__setattr__(self, "_Logger", _logger or __QS_Logger__)

    def model_post_init(self, context: Any, /) -> None:
        self._QS_ID = None
    
    @property
    def QSID(self) -> str:
        """表示对象的全局唯一 id, 且每次运行程序时该 id 不变"""
        if not getattr(self, "_QS_ID", None):
            self._QS_ID = dict2id(self.model_dump())
        return self._QS_ID

    def __setattr__(self, name, value):
        if name in self.model_dump():
            self._QS_ID = None
            if self.Owner: self.Owner._QS_ID = None
        return super().__setattr__(name, value)
    
    def to_dict(self, repr:bool=True) -> dict:
        """以 dict 形式返回所有参数和参数值

        Args:
            repr: 是否仅返回可见参数
        
        Returns:
            {参数名: 参数值}
        """
        if repr:
            return {field: getattr(self, field) for field, info in self.__pydantic_fields__.items() if info.repr}
        else:
            return {field: getattr(self, field) for field in self.__pydantic_fields__.keys()}
    
    def meta(self, key: Optional[Literal["annotation", "title", "description", "default", "required", "frozen", "exclude", "repr"]]=None, repr:bool=True) -> Union[pd.Series, pd.DataFrame]:
        """返回参数集中参数的元信息, 元信息由若干个键值对组成

        Args:
            key: 元信息键, None 表示获取所有的元信息, key 可选下列值
                - annotation: str, 参数值的数据类型
                - title: str, 参数的说明性名称
                - description: str, 参数的描述信息
                - frozen: bool, 参数值初始化后是否不能再修改, 默认值 False
                - exclude: bool, False 表示用于生成参数集的 QSID, 即该参数会影响对象的行为, 默认值 False
                - repr: bool, 该参数是否可见, 默认值 True
            repr: 是否仅返回可见参数
        
        Returns:
            如果 key=None, 则返回 DataFrame(index=[参数名], columns=[所有的 key])
            如果 key 非 None 则返回该 key 对应的元信息, Series(index=[参数名])
        """
        if key is not None:
            return pd.Series({field: getattr(info, key) for field, info in self.__pydantic_fields__.items() if (not repr) or info.repr}, dtype="O")
        else:
            return pd.DataFrame({key: self.meta(key=key, repr=repr) for key in ["annotation", "title", "description", "default", "required", "frozen", "exclude", "repr"]})

    def info(self, arg_names:Optional[List[str]]=None, repr:bool=True, html:bool=False, current_value:bool=True) -> str:
        """返回参数集中参数的说明信息

        Args:
            arg_names: 返回说明信息的参数列表, None 表示返回所有参数
            repr: 是否仅返回可见参数
            html: 是否返回 HTML 格式的说明, False 返回 Markdown 格式的说明
            current_value: 是否返回参数当前取值信息
        
        Returns:
            `参数: 数据类型, 默认值, 描述信息` 格式的列表
        """
        annotation = self.meta(key="annotation", repr=repr)
        title = self.meta(key="title", repr=repr)
        default = self.meta(key="default", repr=repr)
        description = self.meta(key="description", repr=repr)
        key_fmt = "{key}{title}"
        val_fmt = "{annotation}, {default}{description}"
        if current_value: val_fmt += ", 当前取值: {value}"
        if arg_names is None: arg_names = annotation.index
        else: arg_names = annotation.index.intersection(arg_names)
        formatted_info = {}
        for key in arg_names:
            iFormattedKey = key_fmt.format(key=key, title=f"({title[key]})" if title[key] else "")
            iVal = getattr(self, key)
            if isinstance(iVal, __QS_Args__):
                iFormattedVal = "\n"+iVal.info(repr=repr, html=html).replace("\n", "\n    ")
            else:
                iFormattedVal = formatValue2MD(iVal)
            iFormattedVal = val_fmt.format(
                annotation=str(annotation[key]), 
                default="无默认值" if isinstance(default[key], PydanticUndefinedType) else ("默认值 " + (default[key].__repr__() if isinstance(default[key], str) else str(default[key]))), 
                description=(", "+description[key] if description[key] else ""), 
                value=iFormattedVal
            )
            formatted_info[iFormattedKey] = iFormattedVal
        if html:
            return dict2html(formatted_info)
        else:
            return dict2markdown(formatted_info)

    def __getitem__(self, key):
        if not hasattr(self, key):
            raise __QS_Error__(f"参数 '{key}' 不存在, 全体参数为: {list(self.__pydantic_fields__.keys())}")
        return getattr(self, key)

    def __setitem__(self, key, value):
        if not hasattr(self, key):
            self.Logger.warning(f"参数 '{key}' 不存在, 全体参数为: {list(self.__pydantic_fields__.keys())}")
            return
        setattr(self, key, value)

    def __eq__(self, other):
        if not isinstance(other, __QS_Args__): return False
        return self.QSID == other.QSID

    def get(self, key:str, value:Any=None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return value

    def keys(self):
        return self.__pydantic_fields__.keys()

    def values(self):
        return (getattr(self, key) for key in self.__pydantic_fields__.keys())

    def items(self):
        return ((key, getattr(self, key)) for key in self.__pydantic_fields__.keys())

    def update(self, args:dict={}):
        """更新参数集

        Args:
            args: 新的参数和值
        """
        for ifield, ivalue in args.items():
            if ifield in self.__pydantic_fields__:
                setattr(self, ifield, ivalue)

    def __repr__(self):
        if self.Owner:
            return f'{self.Owner.__class__.__name__ if isinstance(self.Owner, __QS_Object__) else str(self.Owner)}.Args({self.__repr_str__(", ")})'
        else:
            return super().__repr__()

    def __str__(self):
        return self.__repr__()

    def _repr_html_(self):
        return dict2html(self.to_dict(), dict_class=(dict, pd.Series), dict_limit=np.inf)

    def serialize(self) -> dict:
        """序列化参数集为 dict，敏感字段自动加密。

        对 ``secret=True`` 的字段值使用 Fernet 加密后以 ``"ENC:<base64>"`` 格式输出，
        其他字段原样输出。

        Returns:
            可 JSON 序列化的 dict
        """
        result = {}
        for field_name, field_info in self.__pydantic_fields__.items():
            value = getattr(self, field_name)
            if _is_secret_field(field_info) and value is not None:
                result[field_name] = encrypt_value(str(value))
            else:
                result[field_name] = value
        return result

    @classmethod
    def deserialize(cls, data: dict) -> "__QS_Args__":
        """从序列化 dict 反向构建参数集实例。

        自动识别 ``"ENC:"`` 前缀并解密对应字段。

        Args:
            data: ``serialize()`` 输出的 dict

        Returns:
            重建的参数集实例
        """
        decrypted = {}
        for field_name, value in data.items():
            if field_name in ("Owner", "Logger"):
                # 兼容旧格式（Owner/Logger 曾是 Field）
                continue
            if is_encrypted(value):
                decrypted[field_name] = decrypt_value(value)
            else:
                decrypted[field_name] = value
        return cls(**decrypted)


class __QS_Object__:
    """Quant Studio 系统对象"""

    __QS_ArgClass__ = __QS_Args__

    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
        """初始化 QuantStudio 系统对象, 参数设置的优先级: args > config_file > 内部默认值

        Args:
            args: 指定的对象参数集
            config_file: 配置文件路径, 配置文件用于设置对象参数。配置文件是一个 json 格式的文件(字符编码为 utf-8, 扩展名为 json), 以键值对的形式给出各个参数的取值
            kwargs:
                logger: 日志对象, 用于打印内部日志, 如果没有指定则使用默认的 QuantStudio.Core.__QS_Logger__ 对象
        """
        self._QS_Logger = kwargs.pop("logger", None)
        if self._QS_Logger is None: self._QS_Logger = __QS_Logger__
        Config = {}
        if config_file:
            if not os.path.isfile(config_file) and (not config_file.startswith(__QS_ConfigPath__)): config_file = __QS_ConfigPath__ + os.sep + config_file
            if os.path.isfile(config_file):
                self._ConfigFile = config_file
                with open(self._ConfigFile, "r", encoding="utf-8") as File:
                    FileStr = File.read()
                    if FileStr: Config = json.loads(FileStr)
            else:
                self._ConfigFile = None
                self._QS_Logger.warning(f"找不到配置文件: {config_file}")
        else:
            self._ConfigFile = None
        args = Config | args
        self._QSArgs = self.__QS_ArgClass__(**args, _owner=self, _logger=self._QS_Logger)
        self._QS_ID = kwargs.get("qs_id", None)

    def model_dump(self) -> Dict[str, Any]:
        return {
            "__type__": "__QS_Object__",
            "__class__": type(self).__module__ + "." + type(self).__qualname__,
            "__qsargs__": self._QSArgs.model_dump()
        }

    @property
    def QSID(self) -> str:
        """表示对象行为的全局唯一 id, 且每次运行程序时该 id 不变。相同 QSID 的对象行为一致，但不同的 QuantStudio 对象有可能 QSID 相同"""
        if getattr(self, "_QS_ID", None) is None:
            self._QS_ID = dict2id(self.model_dump())
        return self._QS_ID

    @property
    def Args(self) -> __QS_Args__:
        """参数集对象"""
        return self._QSArgs

    @property
    def Logger(self):
        """日志对象"""
        return self._QS_Logger

    @property
    def ConfigFile(self) -> str | None:
        """配置文件"""
        if self._ConfigFile:
            return self._ConfigFile
        else:
            return None

    def new(self, args:dict={}, **kwargs) -> "__QS_Object__":
        """给定新的参数集 args 创建一个新的 QuantStudio 对象, args 中未指定的参数则使用原对象的参数

        Args:
            args: 指定的新参数集
            kwargs: 创建 QuantStudio 对象需要的其他入参
        
        Returns:
            QuantStudio 对象
        """
        args = self._QSArgs.model_dump() | args
        kwargs = {"logger": self._QS_Logger, "config_file": self._ConfigFile} | kwargs
        return self.__class__(args=args, **kwargs)
    
    def _repr_html_(self) -> str:
        HTML = f"<b>类</b>: {html.escape(type(self).__module__ + '.' + type(self).__qualname__)}<br/>"
        HTML += f"<b>文档</b>: {html.escape(self.__doc__ if self.__doc__ else '')}<br/>"
        HTML += f"<b>参数</b>: " + self._QSArgs._repr_html_()
        return HTML

    def serialize(self) -> Dict[str, Any]:
        """序列化对象为 dict，敏感字段自动加密。

        调用 ``self.model_dump()`` 获取基础结构，将 ``__qsargs__``
        替换为 ``_QSArgs.serialize()`` 的加密版本。
        将不可 JSON 序列化的子类自定义字段做标记转换。

        Returns:
            可 JSON 序列化的 dict
        """
        result = self.model_dump()
        result["__qsargs__"] = self._QSArgs.serialize()
        # 子类可能覆盖 model_dump 添加额外字段（如 FactorOperator 的 __func__），
        # 此处移除不可序列化的 callable 字段，由子类的 serialize 重新处理。
        for key in list(result):
            if callable(result[key]):
                del result[key]
        return result

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "__QS_Object__":
        """从序列化 dict 反向构建对象实例。

        优先从 ``__class__`` 字段定位目标类（全路径），
        若无法解析则使用调用时的 ``cls``。

        Args:
            data: ``serialize()`` 输出的 dict

        Returns:
            重建的 __QS_Object__ 子类实例
        """
        class_path = data.get("__class__", "")
        if "." in class_path:
            import importlib
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
        qsargs_data = data.get("__qsargs__", {})
        ArgClass = cls.__QS_ArgClass__
        decrypted_args = ArgClass.deserialize(qsargs_data)
        return cls(args=decrypted_args.model_dump())


if __name__ == "__main__":
    class TestArgs(__QS_Args__):
        name: str = Field()# exclude=False, repr=True
        name1: Optional[str] = Field(default=None, exclude=True)
        name2: Optional[str] = Field(default=None, repr=False)

    args = TestArgs(name="aha", name1="aha1", name2="aha2")
    print(args)
    print(args.QSID)

    args.name = "ahaha"
    print(args)
    print(args.QSID)

    print(args.__pydantic_fields__)
    print(args.model_dump())
    print(args.to_dict())
    print(args.to_dict(repr=False))
    # print(args._repr_html_())

    print(args.info(repr=False, html=False))

    print("===")