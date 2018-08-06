# -*- coding: utf-8 -*-
import os
import sys
import operator
import warnings

import numpy as np
import pandas as pd
from traits.api import HasTraits
from traitsui.api import Item, View
from traitsui.menu import OKButton, CancelButton

__QS_MainPath__ = os.path.split(os.path.realpath(__file__))[0]
__QS_CachePath__ = __QS_MainPath__+os.sep+"Cache"
__QS_LibPath__ = __QS_MainPath__+os.sep+"Lib"


for iFileName in os.listdir(path=__QS_CachePath__):
    iFilePath = __QS_CachePath__+os.sep+iFileName
    if os.path.isdir(iFilePath):
        try:
            shutil.rmtree(iFilePath)
        except:
            warnings.warn("缓存文件夹 : '%s' 清除失败!" % iFileName)
    elif os.path.isfile(iFilePath):
        try:
            os.remove(iFilePath)
        except:
            warnings.warn("缓存文件 : '%s' 清除失败!" % iFileName)

# Quant Studio 系统错误
class __QS_Error__(Exception):
    """Quant Studio 错误"""
    pass

# Quant Studio 系统对象
class __QS_Object__(HasTraits):
    """Quant Studio 系统对象"""
    def __init__(self, sys_args={}, **kwargs):
        super().__init__(**kwargs)
        self._LabelTrait = {}
        self._ArgOrder = pd.Series()
        for iTraitName in self.visible_traits():
            iTrait = self.trait(iTraitName)
            if iTrait.arg_type is None: continue
            iLabel = (iTrait.label if iTrait.label is not None else iTraitName)
            iOrder = (iTrait.order if iTrait.order is not None else np.inf)
            self._LabelTrait[iLabel] = iTraitName
            self._ArgOrder[iLabel] = iOrder
        self._ArgOrder.sort_values(inplace=True)
        self.__QS_initArgs__()
        for iArgName, iArgVal in sys_args.items():
            self[iArgName] = iArgVal
        self.trait_view(name="QSView", view_element=View(*self.getViewItems()[0], buttons=[OKButton, CancelButton], resizable=True, title=getattr(self, "Name", "设置参数")))
    @property
    def ArgNames(self):
        return tuple(self._ArgOrder.index)
    def getViewItems(self, context_name=""):
        Prefix = (context_name+"." if context_name else "")
        Context = ({} if not Prefix else {context_name:self})
        return ([Item(Prefix+self._LabelTrait[iLabel]) for iLabel in self._ArgOrder.index], Context)
    def setArgs(self):
        Items, Context = self.getViewItems()
        if Context: return self.configure_traits(view=View(*Items, buttons=[OKButton, CancelButton], resizable=True, title=getattr(self, "Name", "设置参数")), context=Context)
        return self.configure_traits(view=View(*Items, buttons=[OKButton, CancelButton], resizable=True, title=getattr(self, "Name", "设置参数")))
    def add_trait(self, name, *trait):
        if name in self.visible_traits(): return super().add_trait(name, *trait)
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
        iLabel = self.trait(name).label
        Rslt = super().remove_trait(name)
        self._LabelTrait.pop(iLabel)
        self._ArgOrder.pop(iLabel)
        return Rslt
    def __iter__(self):
        return iter(self._LabelTrait)
    def __getitem__(self, key):
        return getattr(self, self._LabelTrait[key])
    def __setitem__(self, key, value):
        setattr(self, self._LabelTrait[key], value)
    def __delitem__(self, key):
        self.remove_trait(self._LabelTrait[key])
    def __QS_initArgs__(self):
        return None

if __name__=="__main__":
    from traits.api import Str
    a = __QS_Object__()
    print(a.ArgNames)
    a.add_trait("aha", Str("aha", arg_type="String"))
    print(a.ArgNames)