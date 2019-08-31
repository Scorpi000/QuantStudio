# -*- coding: utf-8 -*-
import os
import sys
import platform
import logging
import json
import operator
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from traits.api import HasTraits
from traitsui.api import Item, View
from traitsui.menu import OKButton, CancelButton

__QS_MainPath__ = os.path.split(os.path.realpath(__file__))[0]
__QS_LibPath__ = __QS_MainPath__+os.sep+"Lib"
__QS_ConfigPath__ = os.path.expanduser("~")+os.sep+"QuantStudioConfig"

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

# Quant Studio 系统错误
class __QS_Error__(Exception):
    """Quant Studio 错误"""
    pass

# Quant Studio 系统对象
class __QS_Object__(HasTraits):
    """Quant Studio 系统对象"""
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        if "logger" in kwargs: self._QS_Logger = kwargs.pop("logger")
        else: self._QS_Logger = logging.getLogger(__name__)
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
        self._ConfigFile, Config = None, {}
        if config_file and os.path.isfile(config_file):
            self._ConfigFile = config_file
            with open(self._ConfigFile, "r", encoding="utf-8") as File:
                FileStr = File.read()
                if FileStr: Config = json.loads(FileStr)
        Config.update(sys_args)
        for iArgName, iArgVal in Config.items():
            if iArgName in self._ArgOrder.index: self[iArgName] = iArgVal
        self.trait_view(name="QSView", view_element=View(*self.getViewItems()[0], buttons=[OKButton, CancelButton], resizable=True, title=getattr(self, "Name", "设置参数")))
    def __setstate__(self, state, trait_change_notify=False):
        return super().__setstate__(state, trait_change_notify=trait_change_notify)
    @property
    def ArgNames(self):
        return self._ArgOrder.index.tolist()
    @property
    def Args(self):
        return {iArgName:self[iArgName] for iArgName in self.ArgNames}
    @property
    def Logger(self):
        return self._QS_Logger
    def getViewItems(self, context_name=""):
        Prefix = (context_name+"." if context_name else "")
        Context = ({} if not Prefix else {context_name:self})
        return ([Item(Prefix+self._LabelTrait[iLabel]) for iLabel in self._ArgOrder.index], Context)
    def setArgs(self):
        Items, Context = self.getViewItems()
        if Context: return self.configure_traits(view=View(*Items, buttons=[OKButton, CancelButton], resizable=True, title=getattr(self, "Name", "设置参数"), kind="livemodal"), context=Context)
        return self.configure_traits(view=View(*Items, buttons=[OKButton, CancelButton], resizable=True, title=getattr(self, "Name", "设置参数"), kind="livemodal"))
    def getTrait(self, arg_name):
        return (self._LabelTrait[arg_name], self.trait(self._LabelTrait[arg_name]))
    def add_trait(self, name, *trait):
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