# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# 绘制热图
def plotHeatMap(df, ax):
    ax.pcolor(df.values, cmap=matplotlib.cm.Reds)
    ax.set_xticks(np.arange(df.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(df.shape[1])+0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(df.index.astype(str).tolist(), minor=False)
    ax.set_yticklabels(df.columns.astype(str).tolist(), minor=False)
    return ax

# 绘制 K 线图
# quotes: array(shape=(N, 4)), 列分别为开盘价, 最高价, 最低价, 收盘价
def plotCandleStick(ax, quotes, xdata=None, width=0.2, colorup='#B70203', colordown='#3ACCCC', alpha=1.0):
    if xdata is None: xdata = np.arange(quotes.shape[0])
    OFFSET = width / 2.0
    Lines, Patches = [], []
    for i in range(quotes.shape[0]):
        if pd.isnull(quotes[i]).sum()>0: continue
        iOpen, iHigh, iLow, iClose = quotes[i]
        if iClose >= iOpen:
            iColor = colorup
            iLower = iOpen
            iHeight = iClose - iOpen
        else:
            iColor = colordown
            iLower = iClose
            iHeight = iOpen - iClose
        iVLine = Line2D(xdata=(xdata[i], xdata[i]), ydata=(iLow, iHigh), color=iColor, linewidth=0.5, antialiased=True)
        iRect = Rectangle(xy=(i-OFFSET, iLower), width=width, height=iHeight, facecolor=iColor, edgecolor=iColor)
        iRect.set_alpha(alpha)
        Lines.append(iVLine)
        Patches.append(iRect)
        ax.add_line(iVLine)
        ax.add_patch(iRect)
    ax.autoscale_view()
    return Lines, Patches

if __name__=="__main__":
    import datetime as dt
    import matplotlib.pyplot as plt
    import QuantStudio.api as QS
    
    HDB = QS.FactorDB.HDF5DB()
    HDB.connect()
    FT = HDB.getTable("ElementaryFactor")
    DTs = FT.getDateTime(start_dt=dt.datetime(2017,1,1), end_dt=dt.datetime(2017,12,31))
    Price = FT.readData(factor_names=["开盘价", "最高价", "最低价", "收盘价"], ids=["000001.SZ"], dts=DTs).iloc[:, :, 0]
    
    xTicks = np.arange(0, Price.shape[0], max(1, int(Price.shape[0]/10)))
    xTickLabels = [Price.index[i].strftime("%Y-%m-%d") for i in xTicks]

    Fig, Axes = plt.subplots(1, 1, figsize=(16, 8))
    plotCandleStick(Axes, Price.values)
    Axes.set_xticks(xTicks)
    Axes.set_xticklabels(xTickLabels)
    plt.show()
    print("===")