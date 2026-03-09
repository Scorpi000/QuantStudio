# -*- coding: utf-8 -*-
"""截面因子测试"""

from .IC import IC, CalcIC
from .Portfolio import CalcPortfolioNV, CalcMaskPortfolio, MultiPortfolio, makeQuantilePortfolio
from .Correlation import FactorTurnover, SectionCorrelation, CalcFactorTurnover, CalcSectionCorrelation
from .ReturnDecomposition import FamaMacBethRegression, CalcFamaMacBethRegression