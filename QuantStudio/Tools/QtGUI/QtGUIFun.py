# -*- coding: utf-8 -*-
import sys

from PyQt4.QtCore import *
from PyQt4.QtGui import *

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
            table_widget.setItem(jRow, kCol, QTableWidgetItem(str(df.iloc[jRow,kCol])))
    table_widget.blockSignals(False)
    return 0
# 用嵌套字典填充 QTreeWidget
def populateQTreeWidgetWithNestedDict(tree_widget, nested_dict):
    Keys = list(nested_dict.keys())
    Keys.sort()
    for iKey in Keys:
        iValue = nested_dict[iKey]
        iParent = QTreeWidgetItem(tree_widget, [iKey])
        if isinstance(tree_widget, QTreeWidget):
            iParent.setData(0, Qt.UserRole, (iKey,))
        else:
            iParent.setData(0, Qt.UserRole, iParent.parent().data(0,Qt.UserRole)+(iKey,))
        if isinstance(iValue, dict):
            populateQTreeWidgetWithNestedDict(iParent, iValue)
    return 0











# 以 GUI 的方式查看数据集
def showOutput(output, plot_engine="plotly"):
    from QuantStudio.Tools.QtGUI.ResultDlg import PlotlyResultDlg, MatplotlibResultDlg
    App = QApplication(sys.argv)
    if plot_engine=="plotly": Dlg = PlotlyResultDlg(None, output)
    elif plot_engine=="matplotlib": Dlg = MatplotlibResultDlg(None, output)
    Dlg.show()
    App.exec_()
# 以 GUI 的方式设置日期
def setDateTime(dts=[]):
    from QuantStudio.Tools.QtGUI.DateTimeSetup import DateTimeSetupDlg
    App = QApplication(sys.argv)
    Dlg = DateTimeSetupDlg(None, dts)
    Dlg.show()
    App.exec_()
    if Dlg.isChanged: return Dlg.DateTimes
    return dts
# 以 GUI 的方式设置 ID
def setID(ids=[]):
    from QuantStudio.Tools.QtGUI.IDSetup import IDSetupDlg
    App = QApplication(sys.argv)
    Dlg = IDSetupDlg(None, ids)
    Dlg.show()
    App.exec_()
    if Dlg.isChanged: return Dlg.IDs
    return ids