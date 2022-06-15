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
# volume: None 或者 array(shape=(N, )), 成交量
def plotCandleStick(ax, quotes, xdata=None, ax_vol=None, volume=None, colorup='#B70203', colordown='#3ACCCC', rect_width=0.5, line_args={"linewidth": 1, "antialiased": True}, rect_args={}):
    if xdata is None: xdata = np.arange(quotes.shape[0])
    OFFSET = rect_width / 2.0
    PlotVol = ((ax_vol is not None) and (volume is not None))
    Lines, Patches, VolPatches = [], [], []
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
        iVLine = Line2D(xdata=(xdata[i], xdata[i]), ydata=(iLow, iHigh), color=iColor, **line_args)
        iRect = Rectangle(xy=(xdata[i]-OFFSET, iLower), height=iHeight, width=rect_width, facecolor=iColor, edgecolor=iColor, **rect_args)
#         iRect.set_alpha(alpha)
        Lines.append(iVLine)
        Patches.append(iRect)
        ax.add_line(iVLine)
        ax.add_patch(iRect)
        if PlotVol:
            iRect = Rectangle(xy=(xdata[i]-OFFSET, 0), height=volume[i], width=rect_width, facecolor=iColor, edgecolor=iColor, **rect_args)
            VolPatches.append(iRect)
            ax_vol.add_patch(iRect)
    ax.autoscale_view()
    ax_vol.set_ylim((np.nanmin(volume), np.nanmax(volume)))
    ax_vol.autoscale_view()
    if PlotVol:
        return Lines, Patches, VolPatches
    else:
        return Lines, Patches

# 生成时点坐标轴
def setDateTimeAxis(ax, dts, max_display=10, fmt="%Y-%m-%d"):
    nDT = len(dts)
    xTicks = np.arange(0, nDT, max(1, int(nDT/max_display)))
    xTickLabels = [dts[i].strftime(fmt) for i in xTicks]
    ax.set_xticks(xTicks)
    ax.set_xticklabels(xTickLabels)
    return ax

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