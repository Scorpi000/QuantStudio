# -*- coding: utf-8 -*-
import os
import html
import json
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from QuantStudio.Tools.DataTypeConversionFun import dict2html
from QuantStudio.Tools.DataTypeFun import dict2id
from QuantStudio import __QS_ConfigPath__


__QS_Logger__ = logging.getLogger()


# Quant Studio 系统错误
class __QS_Error__(Exception):
    """Quant Studio 错误"""
    pass


# 参数对象
# 参数属性
#   frozen: 初始化后是否可修改, 默认 None 可修改
#   exclude: 参数是否用于生成 ID，默认 False 用于生成
#   repr: 参数是否可见, 默认 True 可见
#   title: 参数名称(对外展示用)
#   description: 描述信息
class QSArgs(BaseModel):
    """参数对象"""
    Owner: Any = Field(default=None, exclude=True, repr=False, frozen=True, title="所有者")
    Logger: Optional[logging.Logger] = Field(default=__QS_Logger__, exclude=True, repr=False, title="日志对象")

    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)

    def model_post_init(self, context: Any, /) -> None:
        self._QS_ID = None

    @property
    def QSID(self):
        if not getattr(self, "_QS_ID", None):
            self._QS_ID = dict2id(self.model_dump())
        return self._QS_ID

    def __setattr__(self, name, value):
        if name in self.model_dump():
            self._QS_ID = None
            if self.Owner: self.Owner._QS_ID = None
        return super().__setattr__(name, value)

    # 以 dict 形式返回所有可见的参数和参数值
    def to_dict(self):
        return {field: getattr(self, field) for field, info in self.__pydantic_fields__.items() if info.repr}

    def __getitem__(self, key):
        if not hasattr(self, key):
            raise __QS_Error__(f"参数 '{key}' 不存在, 全体参数为: {list(self.__pydantic_fields__.keys())}")
        return getattr(self, key)

    def __setitem__(self, key, value):
        if not hasattr(self, key):
            self._QS_Logger.warning(f"参数 '{key}' 不存在, 全体参数为: {list(self.__pydantic_fields__.keys())}")
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
            if not os.path.isfile(config_file): config_file = __QS_ConfigPath__ + os.sep + config_file
            if os.path.isfile(config_file):
                self._ConfigFile = config_file
                with open(self._ConfigFile, "r", encoding="utf-8") as File:
                    FileStr = File.read()
                    if FileStr: Config = json.loads(FileStr)
            else:
                self._ConfigFile = None
                self._QS_Logger.warning("找不到配置文件")
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
    
    def new(self, args={}):
        args = self._QSArgs.model_dump() | args
        return self.__class__(args=args, config_file=self._ConfigFile, logger=self._QS_Logger)
    
    def _repr_html_(self):
        HTML = f"<b>类</b>: {html.escape(str(self.__class__.__name__))}<br/>"
        HTML += f"<b>文档</b>: {html.escape(self.__doc__ if self.__doc__ else '')}<br/>"
        HTML += f"<b>参数</b>: " + self._QSArgs._repr_html_()
        return HTML


if __name__ == "__main__":
    class TestArgs(QSArgs):
        name: str = Field(default=None)# exclude=False, repr=True
        name1: str = Field(default=None, exclude=True)
        name2: str = Field(default=None, repr=False)

    args = TestArgs(name="aha", name1="aha1", name2="aha2")
    print(args)
    print(args.QSID)

    args.name = "ahaha"
    print(args)
    print(args.QSID)

    print(args.__pydantic_fields__)
    print(args.model_dump())
    print(args.to_dict())
    # print(args._repr_html_())

    print("===")