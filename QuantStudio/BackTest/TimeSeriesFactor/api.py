# -*- coding: utf-8 -*-
"""时序因子测试"""

from .Difference import QuantileDifference
from .Correlation import TimeSeriesCorrelation
from .ReturnForecast import ReturnForecast, OLS
from .Spread import Cointegration
from .Timing import QuantileTiming, TargetPositionSignal, TradeSignal