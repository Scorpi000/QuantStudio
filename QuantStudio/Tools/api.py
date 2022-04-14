# -*- coding: utf-8 -*-
from . import MathFun as Math
from . import DateTimeFun as DateTime
from . import StrategyTestFun as Strategy
from . import DataPreprocessingFun as Preprocess
from . import FileFun as File
try:
    from .QtGUI import QtGUIFun as QtGUI
except:
    print("Qt GUI 工具导入失败!")
from .AuxiliaryFun import genAvailableName

import pandas as pd
if pd.__version__<'0.25.0':
    Panel = pd.Panel
    #from QuantStudio.Tools.QSObjects import Panel
else:
    from QuantStudio.Tools.QSObjects import Panel