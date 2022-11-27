# -*- coding: utf-8 -*-
"""截面因子测试"""

from .IC import IC, RiskAdjustedIC, ICDecay
from .Portfolio import QuantilePortfolio, FilterPortfolio, MultiPortfolio
from .Distribution import IndustryDistribution
from .Correlation import FactorTurnover, SectionCorrelation
from .ReturnDecomposition import FamaMacBethRegression