# coding=utf-8
import numpy as np
import pandas as pd
 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdate

yData1 = np.random.randn(10)
yData2 = np.random.randn(10)
xData = np.arange(0, yData1.shape[0])
xTickLabels = [str(int(iInd))+"G" for iInd in xData]
plt.bar(xData, yData1, width=-0.25, align="edge", color="r")
plt.bar(xData, yData2, width=0.25, align="edge", color="b")
plt.gca().set_xticks(xData)
plt.gca().set_xticklabels(xTickLabels)
plt.show()