# -*- coding: utf-8 -*-
from . import MathFun as Math
from . import DateTimeFun as DateTime
from . import StrategyTestFun as Strategy
from . import DataPreprocessingFun as Preprocess
from . import FileFun as File
try:
    from .QtGUI import QtGUIFun as QtGUI
except:
    print("PyQt4 未安装, 部分功能无法使用, 如有需要请自行安装 PyQt4!")
from .AuxiliaryFun import genAvailableName