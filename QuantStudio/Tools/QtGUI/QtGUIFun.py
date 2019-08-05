# -*- coding: utf-8 -*-
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

App = QtWidgets.QApplication(sys.argv)

# 用 DataFrame 填充 QTableWidget, 数据形式
def populateTableWithDataFrame(table_widget, df):
    table_widget.blockSignals(True)
    table_widget.clear()
    nRow,nCol = df.shape
    ColumnLabels = [str(iLabel) for iLabel in df.columns]
    table_widget.setColumnCount(nCol)
    table_widget.setHorizontalHeaderLabels(ColumnLabels)
    RowLabels = [str(iLabel) for iLabel in df.index]
    table_widget.setRowCount(nRow)
    table_widget.setVerticalHeaderLabels(RowLabels)
    for jRow in range(nRow):
        for kCol in range(nCol):
            table_widget.setItem(jRow, kCol, QtWidgets.QTableWidgetItem(str(df.iloc[jRow,kCol])))
    table_widget.blockSignals(False)
    return 0
# 用嵌套字典填充 QTreeWidget
def populateQTreeWidgetWithNestedDict(tree_widget, nested_dict):
    Keys = list(nested_dict.keys())
    Keys.sort()
    for iKey in Keys:
        iValue = nested_dict[iKey]
        iParent = QtWidgets.QTreeWidgetItem(tree_widget, [iKey])
        if isinstance(tree_widget, QtWidgets.QTreeWidget):
            iParent.setData(0, QtCore.Qt.UserRole, (iKey,))
        else:
            iParent.setData(0, QtCore.Qt.UserRole, iParent.parent().data(0, QtCore.Qt.UserRole)+(iKey,))
        if isinstance(iValue, dict):
            populateQTreeWidgetWithNestedDict(iParent, iValue)
    return 0

# 以 GUI 的方式查看数据集
def showOutput(output, plot_engine="matplotlib"):
    from QuantStudio.Tools.QtGUI.ResultDlg import PlotlyResultDlg, MatplotlibResultDlg
    if plot_engine=="plotly": Dlg = PlotlyResultDlg(None, output)
    elif plot_engine=="matplotlib": Dlg = MatplotlibResultDlg(None, output)
    Dlg.show()
    App.exec_()
    return 0
# 以 GUI 的方式查看因子库
def showFactorDB(fdb):
    from QuantStudio.Tools.QtGUI.FactorDBDlg import FactorDBDlg
    Dlg = FactorDBDlg(fdb)
    Dlg.show()
    App.exec_()
    return 0
# 以 GUI 的方式查看因子
def showFactor(factor):
    from QuantStudio.Tools.QtGUI.PreviewFactorDlg import PreviewDlg
    Dlg = PreviewDlg(factor)
    Dlg.show()
    App.exec_()
    return 0
# 以 GUI 的方式查看风险库
def showRiskDB(rdb):
    from QuantStudio.Tools.QtGUI.RiskDBDlg import RiskDBDlg
    Dlg = RiskDBDlg(rdb)
    Dlg.show()
    App.exec_()
    return 0
# 以 GUI 的方式设置日期时间
def setDateTime(dts=[], dates=[], times=[], ft=None):
    from QuantStudio.Tools.QtGUI.DateTimeSetup import DateTimeSetupDlg
    Dlg = DateTimeSetupDlg(dts=dts, dates=dates, times=times, ft=ft)
    Dlg.show()
    App.exec_()
    if Dlg.isChanged: return (Dlg.DateTimes, Dlg.Dates, Dlg.Times)
    else: return (dts, dates, times)
# 以 GUI 的方式设置 ID
def setID(ids=[], ft=None):
    from QuantStudio.Tools.QtGUI.IDSetup import IDSetupDlg
    Dlg = IDSetupDlg(ids=ids, ft=ft)
    Dlg.show()
    App.exec_()
    if Dlg.isChanged: return Dlg.IDs
    else: return ids