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
from QuantStudio.Tools.DataTypeConversionFun import dict2html, dict2markdown
from QuantStudio.Tools.DataTypeFun import dict2id


__QS_Logger__ = logging.getLogger('QS')
__QS_Logger__.setLevel(logging.INFO)
_QSLogHandler = logging.StreamHandler()
_QSLogHandler.setLevel(logging.INFO)
_QSLogHandler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s : %(message)s'))
__QS_Logger__.addHandler(_QSLogHandler)


# Quant Studio 系统错误
class __QS_Error__(Exception):
    """Quant Studio 错误"""
    pass


# 参数对象
# 参数属性
#   frozen: 初始化后是否可修改, 默认 None 可修改
#   exclude: 参数是否不用于生成 ID，默认 False 用于生成
#   repr: 参数是否可见, 默认 True 可见
#   title: 参数名称(对外展示用)
#   description: 描述信息
class QSArgs(BaseModel):
    """参数对象"""
    Owner: Any = Field(default=None, exclude=True, repr=False, frozen=True, title="所有者")
    Logger: logging.Logger = Field(default=__QS_Logger__, exclude=True, repr=False, title="日志对象")

    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)
    
    def model_post_init(self, context: Any, /) -> None:
        self._QS_ID = None
    
    @property
    def QSID(self) -> str:
        if not getattr(self, "_QS_ID", None):
            self._QS_ID = dict2id(self.model_dump())
        return self._QS_ID

    def __setattr__(self, name, value):
        if name in self.model_dump():
            self._QS_ID = None
            if self.Owner: self.Owner._QS_ID = None
        return super().__setattr__(name, value)

    # 以 dict 形式返回所有参数和参数值, repr=True: 仅返回可见参数
    def to_dict(self, repr=True) -> dict:
        if repr:
            return {field: getattr(self, field) for field, info in self.__pydantic_fields__.items() if info.repr}
        else:
            return {field: getattr(self, field) for field in self.__pydantic_fields__.keys()}
    
    # 给定 key，返回参数集中参数的元信息
    def meta(self, key: Optional[Literal["annotation", "title", "description", "default", "required", "frozen", "exclude", "repr"]]=None, repr=True) -> Union[pd.Series, pd.DataFrame]:
        if key is not None:
            return pd.Series({field: getattr(info, key) for field, info in self.__pydantic_fields__.items() if (not repr) or info.repr}, dtype="O")
        else:
            return pd.DataFrame({key: self.meta(key=key, repr=repr) for key in ["annotation", "title", "description", "default", "required", "frozen", "exclude", "repr"]})

    # 返回参数的说明信息
    def info(self, repr=True, html=False) -> str:
        annotation = self.meta(key="annotation", repr=repr)
        title = self.meta(key="title", repr=repr)
        default = self.meta(key="default", repr=repr)
        description = self.meta(key="description", repr=repr)
        key_fmt = "{key}{title}"
        val_fmt = "{annotation}, {default}{description}"
        formatted_info = {key_fmt.format(key=key, title=f"({title[key]})" if title[key] else ""): val_fmt.format(annotation=str(annotation[key]), default="无默认值" if isinstance(default[key], PydanticUndefinedType) else "默认值 "+str(default[key]), description=(", "+description[key] if description[key] else "")) for key in annotation.index}
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
        if not isinstance(other, QSArgs): return False
        return self.QSID == other.QSID

    def get(self, key:str, value=None):
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

    def update(self, args:dict={}) -> None:
        for ifield, ivalue in args.items():
            if ifield in self.__pydantic_fields__:
                setattr(self, ifield, ivalue)

    def __repr__(self):
        if self.Owner:
            return f'{self.Owner.__class__.__name__ if isinstance(self.Owner, __QS_Object__) else str(self.Owner)}.QSArgs({self.__repr_str__(", ")})'
        else:
            return super().__repr__()

    def __str__(self):
        return self.__repr__()

    def _repr_html_(self):
        return dict2html(self.to_dict(), dict_class=(dict, pd.Series), dict_limit=np.inf)


# Quant Studio 系统对象
class __QS_Object__:
    """Quant Studio 系统对象"""
    __QS_ArgClass__ = QSArgs

    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
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
    def QSID(self):
        if not getattr(self, "_QS_ID", None):
            self._QS_ID = dict2id(self.model_dump())
        return self._QS_ID

    @property
    def Args(self):
        return self._QSArgs

    @property
    def Logger(self):
        return self._QS_Logger
    
    def new(self, args={}, **kwargs):
        args = self._QSArgs.model_dump() | args
        kwargs = {"logger": self._QS_Logger, "config_file": self._ConfigFile} | kwargs
        return self.__class__(args=args, **kwargs)
    
    def _repr_html_(self):
        HTML = f"<b>类</b>: {html.escape(str(self.__class__.__name__))}<br/>"
        HTML += f"<b>文档</b>: {html.escape(self.__doc__ if self.__doc__ else '')}<br/>"
        HTML += f"<b>参数</b>: " + self._QSArgs._repr_html_()
        return HTML


if __name__ == "__main__":
    class TestArgs(QSArgs):
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