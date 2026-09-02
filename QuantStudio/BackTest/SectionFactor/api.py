# -*- coding: utf-8 -*-
"""截面因子测试"""

from .IC import IC, CalcIC, ICReport, ICDecayReport
from .QuantilePortfolio import MultiPortfolio, makeQuantilePortfolio, MultiPortfolioReport
from .Correlation import FactorTurnover, SectionCorrelation, CalcFactorTurnover, CalcSectionCorrelation, FactorTurnoverReport, SectionCorrelationReport
from .ReturnDecomposition import FamaMacBethRegression, CalcFamaMacBethRegression, FamaMacBethRegressionReport