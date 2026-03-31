# -*- coding: utf-8 -*-
import os
import html
import json
import logging
from typing import Any, Optional, Literal, Union

import numpy as np
import pandas as pd
from pydantic_core import PydanticUndefinedType
from pydantic import BaseModel, ConfigDict, Field

from QuantStudio import __QS_ConfigPath__
from QuantStudio.Tools.DataTypeConversionFun import dict2html, dict2markdown, formatValue2MD
from QuantStudio.Tools.DataTypeFun import dict2id


def setDefaultLogLevel(level=logging.INFO):
    global __QS_Logger__
    __QS_Logger__ = logging.getLogger('QS')
    __QS_Logger__.setLevel(level)
    for iHandler in __QS_Logger__.handlers:
        __QS_Logger__.removeHandler(iHandler)
    _QSLogHandler = logging.StreamHandler()
    _QSLogHandler.setLevel(level)
    _QSLogHandler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s : %(message)s'))
    __QS_Logger__.addHandler(_QSLogHandler)

def setDefaultLogger(logger):
    global __QS_Logger__
    __QS_Logger__ = logger

__QS_Logger__ = None
setDefaultLogLevel()

class __QS_Error__(Exception):
    """Quant Studio 系统错误"""
    pass


class __QS_Args__(BaseModel):
    """QuantStudio 参数对象"""

    Owner: Any = Field(default=None, exclude=True, repr=False, frozen=True, title="所有者")
    Logger: logging.Logger = Field(default=__QS_Logger__, exclude=True, repr=False, title="日志对象")
    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)
    
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

    def info(self, repr:bool=True, html:bool=False) -> str:
        """返回参数集中参数的说明信息

        Args:
            repr: 是否仅返回可见参数
            html: 是否返回 HTML 格式的说明, False 返回 Markdown 格式的说明
        
        Returns:
            `参数: 数据类型, 默认值, 描述信息` 格式的列表
        """
        annotation = self.meta(key="annotation", repr=repr)
        title = self.meta(key="title", repr=repr)
        default = self.meta(key="default", repr=repr)
        description = self.meta(key="description", repr=repr)
        key_fmt = "{key}{title}"
        val_fmt = "{annotation}, {default}{description}, 当前取值: {value}"
        formatted_info = {}
        for key in annotation.index:
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
        args = Config | args | {"Owner": self, "Logger": self._QS_Logger}
        self._QSArgs = self.__QS_ArgClass__(**args)

    def model_dump(self):
        return {
            "__type__": "__QS_Object__",
            "__class__": self.__class__.__name__,
            "__qsargs__": self._QSArgs.model_dump()
        }

    @property
    def QSID(self) -> str:
        """表示对象行为的全局唯一 id, 且每次运行程序时该 id 不变。相同 QSID 的对象行为一致，但不同的 QuantStudio 对象有可能 QSID 相同"""
        if not getattr(self, "_QS_ID", None):
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
    
    def _repr_html_(self):
        HTML = f"<b>类</b>: {html.escape(str(self.__class__.__name__))}<br/>"
        HTML += f"<b>文档</b>: {html.escape(self.__doc__ if self.__doc__ else '')}<br/>"
        HTML += f"<b>参数</b>: " + self._QSArgs._repr_html_()
        return HTML


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