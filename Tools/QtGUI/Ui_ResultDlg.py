# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Project\Python\QuantStudio\GUI\ResultDlg.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_ResultDlg(object):
    def setupUi(self, ResultDlg):
        ResultDlg.setObjectName(_fromUtf8("ResultDlg"))
        ResultDlg.resize(925, 644)
        ResultDlg.setSizeGripEnabled(True)
        self.horizontalLayout = QtGui.QHBoxLayout(ResultDlg)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.MainResultTable = QtGui.QTableWidget(ResultDlg)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.MainResultTable.sizePolicy().hasHeightForWidth())
        self.MainResultTable.setSizePolicy(sizePolicy)
        self.MainResultTable.setMinimumSize(QtCore.QSize(731, 0))
        self.MainResultTable.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.MainResultTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.MainResultTable.setSelectionBehavior(QtGui.QAbstractItemView.SelectColumns)
        self.MainResultTable.setObjectName(_fromUtf8("MainResultTable"))
        self.MainResultTable.setColumnCount(0)
        self.MainResultTable.setRowCount(0)
        self.horizontalLayout.addWidget(self.MainResultTable)
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label = QtGui.QLabel(ResultDlg)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 3)
        self.GenTableButton = QtGui.QPushButton(ResultDlg)
        self.GenTableButton.setObjectName(_fromUtf8("GenTableButton"))
        self.gridLayout.addWidget(self.GenTableButton, 4, 0, 1, 1)
        self.TransposeButton = QtGui.QPushButton(ResultDlg)
        self.TransposeButton.setObjectName(_fromUtf8("TransposeButton"))
        self.gridLayout.addWidget(self.TransposeButton, 4, 1, 1, 2)
        self.PlotButton = QtGui.QPushButton(ResultDlg)
        self.PlotButton.setObjectName(_fromUtf8("PlotButton"))
        self.gridLayout.addWidget(self.PlotButton, 5, 0, 1, 1)
        self.ExportButton = QtGui.QPushButton(ResultDlg)
        self.ExportButton.setObjectName(_fromUtf8("ExportButton"))
        self.gridLayout.addWidget(self.ExportButton, 5, 1, 1, 2)
        self.RowLimitCheckBox = QtGui.QCheckBox(ResultDlg)
        self.RowLimitCheckBox.setChecked(False)
        self.RowLimitCheckBox.setObjectName(_fromUtf8("RowLimitCheckBox"))
        self.gridLayout.addWidget(self.RowLimitCheckBox, 2, 0, 1, 1)
        self.RowLimitSpinBox = QtGui.QSpinBox(ResultDlg)
        self.RowLimitSpinBox.setMaximum(9999)
        self.RowLimitSpinBox.setProperty("value", 100)
        self.RowLimitSpinBox.setObjectName(_fromUtf8("RowLimitSpinBox"))
        self.gridLayout.addWidget(self.RowLimitSpinBox, 2, 1, 1, 2)
        self.ColumnLimitCheckBox = QtGui.QCheckBox(ResultDlg)
        self.ColumnLimitCheckBox.setChecked(True)
        self.ColumnLimitCheckBox.setObjectName(_fromUtf8("ColumnLimitCheckBox"))
        self.gridLayout.addWidget(self.ColumnLimitCheckBox, 3, 0, 1, 1)
        self.ColumnLimitSpinBox = QtGui.QSpinBox(ResultDlg)
        self.ColumnLimitSpinBox.setMaximum(9999)
        self.ColumnLimitSpinBox.setProperty("value", 20)
        self.ColumnLimitSpinBox.setObjectName(_fromUtf8("ColumnLimitSpinBox"))
        self.gridLayout.addWidget(self.ColumnLimitSpinBox, 3, 1, 1, 2)
        self.MainResultTree = QtGui.QTreeWidget(ResultDlg)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.MainResultTree.sizePolicy().hasHeightForWidth())
        self.MainResultTree.setSizePolicy(sizePolicy)
        self.MainResultTree.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.MainResultTree.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.MainResultTree.setObjectName(_fromUtf8("MainResultTree"))
        self.gridLayout.addWidget(self.MainResultTree, 1, 0, 1, 3)
        self.horizontalLayout.addLayout(self.gridLayout)
        self.label.setBuddy(self.MainResultTable)

        self.retranslateUi(ResultDlg)
        QtCore.QMetaObject.connectSlotsByName(ResultDlg)
        ResultDlg.setTabOrder(self.MainResultTree, self.GenTableButton)
        ResultDlg.setTabOrder(self.GenTableButton, self.MainResultTable)

    def retranslateUi(self, ResultDlg):
        ResultDlg.setWindowTitle(_translate("ResultDlg", "测试结果", None))
        self.label.setText(_translate("ResultDlg", "<html><head/><body><p align=\"center\">主要结果</p></body></html>", None))
        self.GenTableButton.setText(_translate("ResultDlg", "<<", None))
        self.TransposeButton.setText(_translate("ResultDlg", "转置", None))
        self.PlotButton.setText(_translate("ResultDlg", "绘制图像", None))
        self.ExportButton.setText(_translate("ResultDlg", "导出数据", None))
        self.RowLimitCheckBox.setText(_translate("ResultDlg", "显示行数", None))
        self.ColumnLimitCheckBox.setText(_translate("ResultDlg", "显示列数", None))
        self.MainResultTree.headerItem().setText(0, _translate("ResultDlg", "变量", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    ResultDlg = QtGui.QDialog()
    ui = Ui_ResultDlg()
    ui.setupUi(ResultDlg)
    ResultDlg.show()
    sys.exit(app.exec_())

