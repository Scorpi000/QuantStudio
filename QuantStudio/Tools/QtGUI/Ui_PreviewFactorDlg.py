# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Project\QuantStudio\QuantStudio\Tools\QtGUI\PreviewFactorDlg.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_PreviewDlg(object):
    def setupUi(self, PreviewDlg):
        PreviewDlg.setObjectName("PreviewDlg")
        PreviewDlg.resize(829, 512)
        PreviewDlg.setSizeGripEnabled(True)
        self.gridLayout = QtWidgets.QGridLayout(PreviewDlg)
        self.gridLayout.setObjectName("gridLayout")
        self.PreviewTable = QtWidgets.QTableWidget(PreviewDlg)
        self.PreviewTable.setMinimumSize(QtCore.QSize(631, 0))
        self.PreviewTable.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked)
        self.PreviewTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.PreviewTable.setObjectName("PreviewTable")
        self.PreviewTable.setColumnCount(0)
        self.PreviewTable.setRowCount(0)
        self.PreviewTable.verticalHeader().setDefaultSectionSize(30)
        self.gridLayout.addWidget(self.PreviewTable, 0, 0, 6, 1)
        self.DateTimeButton = QtWidgets.QPushButton(PreviewDlg)
        font = QtGui.QFont()
        font.setFamily("Aharoni")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.DateTimeButton.setFont(font)
        self.DateTimeButton.setObjectName("DateTimeButton")
        self.gridLayout.addWidget(self.DateTimeButton, 0, 1, 1, 2)
        self.DateTimeList = QtWidgets.QListWidget(PreviewDlg)
        self.DateTimeList.setObjectName("DateTimeList")
        self.gridLayout.addWidget(self.DateTimeList, 1, 1, 1, 2)
        self.IDButton = QtWidgets.QPushButton(PreviewDlg)
        font = QtGui.QFont()
        font.setFamily("Aharoni")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.IDButton.setFont(font)
        self.IDButton.setObjectName("IDButton")
        self.gridLayout.addWidget(self.IDButton, 2, 1, 1, 2)
        self.IDList = QtWidgets.QListWidget(PreviewDlg)
        self.IDList.setObjectName("IDList")
        self.gridLayout.addWidget(self.IDList, 3, 1, 1, 2)
        self.ArgSetButton = QtWidgets.QPushButton(PreviewDlg)
        font = QtGui.QFont()
        font.setFamily("Aharoni")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.ArgSetButton.setFont(font)
        self.ArgSetButton.setObjectName("ArgSetButton")
        self.gridLayout.addWidget(self.ArgSetButton, 4, 1, 1, 1)
        self.ExecuteButton = QtWidgets.QPushButton(PreviewDlg)
        font = QtGui.QFont()
        font.setFamily("Aharoni")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.ExecuteButton.setFont(font)
        self.ExecuteButton.setObjectName("ExecuteButton")
        self.gridLayout.addWidget(self.ExecuteButton, 4, 2, 1, 1)
        self.Export2CSVButton = QtWidgets.QPushButton(PreviewDlg)
        font = QtGui.QFont()
        font.setFamily("Aharoni")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.Export2CSVButton.setFont(font)
        self.Export2CSVButton.setObjectName("Export2CSVButton")
        self.gridLayout.addWidget(self.Export2CSVButton, 5, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(90, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 5, 2, 1, 1)

        self.retranslateUi(PreviewDlg)
        QtCore.QMetaObject.connectSlotsByName(PreviewDlg)
        PreviewDlg.setTabOrder(self.ExecuteButton, self.PreviewTable)

    def retranslateUi(self, PreviewDlg):
        _translate = QtCore.QCoreApplication.translate
        PreviewDlg.setWindowTitle(_translate("PreviewDlg", "因子预览"))
        self.DateTimeButton.setText(_translate("PreviewDlg", "时点"))
        self.IDButton.setText(_translate("PreviewDlg", "ID"))
        self.ArgSetButton.setText(_translate("PreviewDlg", "参数"))
        self.ExecuteButton.setText(_translate("PreviewDlg", "预览"))
        self.Export2CSVButton.setText(_translate("PreviewDlg", "导出CSV"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    PreviewDlg = QtWidgets.QDialog()
    ui = Ui_PreviewDlg()
    ui.setupUi(PreviewDlg)
    PreviewDlg.show()
    sys.exit(app.exec_())

