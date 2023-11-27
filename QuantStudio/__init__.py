# -*- coding: utf-8 -*-
import os
import html
import platform
import logging
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from traits.api import HasTraits

__QS_MainPath__ = os.path.split(os.path.realpath(__file__))[0]
__QS_LibPath__ = __QS_MainPath__+os.sep+"Lib"
__QS_ConfigPath__ = os.path.expanduser("~")+os.sep+"QuantStudioConfig"
__QS_Logger__ = logging.getLogger()

from matplotlib.pylab import mpl
if platform.system()=="Windows":
    mpl.rcParams['font.sans-serif'] = ["SimHei"]
elif platform.system()=="Darwin":
    if os.path.isfile("/Library/Fonts/Arial Unicode.ttf"):
        from matplotlib.font_manager import FontProperties
        Font = FontProperties(fname="/Library/Fonts/Arial Unicode.ttf")
        mpl.rcParams["font.family"] = Font.get_family()
        mpl.rcParams["font.sans-serif"] = Font.get_name()
mpl.rcParams['axes.unicode_minus'] = False

from QuantStudio.Tools.DataTypeConversionFun import dict2html

# 参数对象
# trait 的附加属性
#     label: 参数名称(对外使用)
#     order: 参数的排序
#     arg_type: 参数的类型: "String", "Integer", "Bool", "Float", "Dict", "List", "File", "Directory", "ArgObject"
#     mutable: 初始化后是否可修改, 默认 None 可修改
#     eq_arg: 是否用于判断两个参数对象相等, 默认 None 用于判断
class QSArgs(HasTraits):
    """参数对象"""
    def __init__(self, owner=None, sys_args={}, config_file=None, **kwargs):
        self._QS_Frozen = False# 是否冻结参数, 不允许增删参数, 对于 mutable=False 的参数不允许修改值
        self._QS_Logger = kwargs.pop("logger", None)
        if self._QS_Logger is None: self._QS_Logger = __QS_Logger__
        super().__init__(**kwargs)
        self._Owner = owner
        self._LabelTrait = {}
        self._ArgOrder = pd.Series(dtype=float)
        for iTraitName in self.visible_traits():
            iTrait = self.trait(iTraitName)
            if iTrait.arg_type is None: continue
            iLabel = (iTrait.label if iTrait.label is not None else iTraitName)
            iOrder = (iTrait.order if iTrait.order is not None else np.inf)
            self._LabelTrait[iLabel] = iTraitName
            self._ArgOrder[iLabel] = iOrder
        self._ArgOrder.sort_values(inplace=True)
        self.__QS_initArgs__()
        self._ConfigFile, Config = None, {}
        if config_file:
            if not os.path.isfile(config_file): config_file = __QS_ConfigPath__+os.sep+config_file
            if os.path.isfile(config_file):
                self._ConfigFile = config_file
                with open(self._ConfigFile, "r", encoding="utf-8") as File:
                    FileStr = File.read()
                    if FileStr: Config = json.loads(FileStr)
        Config.update(sys_args)
        for iArgName, iArgVal in Config.items():
            if iArgName in self._ArgOrder.index: self[iArgName] = iArgVal
        self._QS_Frozen = True
        
    def __setstate__(self, state, trait_change_notify=False):
        return super().__setstate__(state, trait_change_notify=trait_change_notify)
    
    def __str__(self):
        return str(self.to_dict())
    
    def __repr__(self):
        return repr(self.to_dict())
    
    @property
    def ArgNames(self):
        return self._ArgOrder.index.tolist()
    
    @property
    def ObservedArgs(self):
        return ()

    @property
    def Owner(self):
        return self._Owner
    
    @property
    def Logger(self):
        return self._QS_Logger
    
    def to_dict(self):
        return {iArgName:self[iArgName] for iArgName in self.ArgNames}

    def copy(self):
        return self.to_dict()

    def getTrait(self, arg_name):
        return (self._LabelTrait[arg_name], self.trait(self._LabelTrait[arg_name]))
    
    def add_trait(self, name, *trait):
        if self._QS_Frozen and any(iTrait.arg_type is not None for iTrait in trait):
            raise __QS_Error__(f"参数集已冻结, 不能增加参数 '{name}'")
        Rslt = super().add_trait(name, *trait)
        iTrait = self.trait(name)
        if iTrait.arg_type is None: return Rslt
        iLabel = (iTrait.label if iTrait.label is not None else name)
        iOrder = (iTrait.order if iTrait.order is not None else np.inf)
        self._LabelTrait[iLabel] = name
        self._ArgOrder[iLabel] = iOrder
        self._ArgOrder.sort_values(inplace=True)
        return Rslt
    
    def remove_trait(self, name):
        if (name not in self.visible_traits()) or (self.trait(name).arg_type is None): return super().remove_trait(name)
        if self._QS_Frozen:
            raise __QS_Error__(f"参数集已冻结, 不能删除参数 '{name}'")
        iLabel = self.trait(name).label
        Rslt = super().remove_trait(name)
        self._LabelTrait.pop(iLabel)
        self._ArgOrder.pop(iLabel)
        return Rslt
    
    def __iter__(self):
        return iter(self._LabelTrait)
    
    def __len__(self):
        return len(self._LabelTrait)

    def __getitem__(self, key):
        return getattr(self, self._LabelTrait[key])
    
    def __setitem__(self, key, value):
        iTrait = self.trait(self._LabelTrait[key])
        iMutable = (True if iTrait.mutable is None else iTrait.mutable)
        if self._QS_Frozen and (not iMutable):
            raise __QS_Error__(f"参数集已冻结且参数 '{key}' 是不可变参数, 不能修改")
        if iTrait.arg_type == "ArgObject":
            iArgObj = self[key]
            for iKey, iVal in value.items():
                iArgObj[iKey] = iVal
        else:
            setattr(self, self._LabelTrait[key], value)
    
    def __delitem__(self, key):
        if self._QS_Frozen:
            raise __QS_Error__(f"参数集已冻结, 不能删除参数 '{key}'")
        self.remove_trait(self._LabelTrait[key])

    def __contains__(self, key):
        return (key in self._LabelTrait)
    
    def __eq__(self, other):
        if not isinstance(other, QSArgs): return False
        try:
            for iArgName, iTraitName in self._LabelTrait.items():
                iTrait = self.trait(iTraitName)
                iOtherTrait = other.trait(iTraitName)
                if iOtherTrait is None: return False
                iEqArg, iOtherEqArg = bool(iTrait.eq_arg is None or iTrait.eq_arg), bool(iOtherTrait.eq_arg is None or iOtherTrait.eq_arg)
                if iEqArg and iOtherEqArg:
                    iVal, iOtherVal = getattr(self, iTraitName), getattr(other, iTraitName)
                    if iVal is iOtherVal: continue
                    if not (isinstance(iOtherVal, type(iVal)) or isinstance(iVal, type(iOtherVal))): return False
                    if isinstance(iVal, (pd.DataFrame, pd.Series)) and (not iVal.equals(iOtherVal)): return False
                    if isinstance(iVal, (np.ndarray, np.matrix)) and (not pd.DataFrame(iVal).equals(pd.DataFrame(iOtherVal))): return False
                    if isinstance(iVal, __QS_Object__) and (iVal is not iOtherVal): return False
                    if iVal!=iOtherVal: return False
                elif (not iEqArg) and (not iOtherEqArg):
                    continue
                else:
                    return False
        except Exception as e:
            self._QS_Logger.warning(f"参数集 {self} 和 {other} 确定是否相等时错误: {e}")
            return False
        else:
            return True

    def get(self, key, value=None):
        if key in self._LabelTrait:
            return getattr(self, self._LabelTrait[key])
        else:
            return value
    
    def keys(self):
        return tuple(self._ArgOrder.index)
    
    def values(self):
        return (getattr(self, self._LabelTrait[iKey]) for iKey in self._ArgOrder.index)
    
    def items(self):
        return zip(self.keys(), self.values())
    
    def update(self, args={}):
        for iKey in self._ArgOrder.index.intersection(args.keys()):
            self[iKey] = args[iKey]
    
    def clear(self):
        for iArgName in self._ArgOrder.index:
            iKey = self._LabelTrait[iArgName]
            iTrait = self.trait(iKey)
            iMutable = (True if iTrait.mutable is None else iTrait.mutable)
            if iMutable:
                setattr(self, iKey, iTrait.default)
    
    def __QS_initArgs__(self):
        return None

    def _repr_html_(self):
        return dict2html(self, dict_class=(dict, pd.Series, QSArgs), dict_limit=np.inf)
    
    
# Quant Studio 系统错误
class __QS_Error__(Exception):
    """Quant Studio 错误"""
    pass

# Quant Studio 系统对象
class __QS_Object__:
    """Quant Studio 系统对象"""
    __QS_ArgClass__ = QSArgs
    
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._QS_Logger = kwargs.pop("logger", None)
        if self._QS_Logger is None: self._QS_Logger = __QS_Logger__
        self._QSArgs = self.__QS_ArgClass__(owner=self, sys_args=sys_args, config_file=config_file, logger=self._QS_Logger)
    
    @property
    def Logger(self):
        return self._QS_Logger
    
    @property
    def Args(self):
        return self._QSArgs
    
    def _repr_html_(self):
        HTML = f"<b>类</b>: {html.escape(str(self.__class__))}<br/>"
        HTML += f"<b>文档</b>: {html.escape(self.__doc__ if self.__doc__ else '')}<br/>"
        HTML += f"<b>参数</b>: " + self._QSArgs._repr_html_()
        return HTML

if __name__=="__main__":
    import QuantStudio.api as QS
    HDB1 = QS.FactorDB.HDF5DB()
    HDB2 = QS.FactorDB.HDF5DB()
    print(HDB1.Args == HDB2.Args)