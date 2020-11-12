# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Project\QuantStudio\QuantStudio\Tools\QtGUI\FactorDBDlg.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_FactorDBDlg(object):
    def setupUi(self, FactorDBDlg):
        FactorDBDlg.setObjectName("FactorDBDlg")
        FactorDBDlg.resize(412, 539)
        FactorDBDlg.setSizeGripEnabled(True)
        self.gridLayout = QtWidgets.QGridLayout(FactorDBDlg)
        self.gridLayout.setObjectName("gridLayout")
        self.FactorDBTree = QtWidgets.QTreeWidget(FactorDBDlg)
        self.FactorDBTree.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.FactorDBTree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.FactorDBTree.setObjectName("FactorDBTree")
        self.gridLayout.addWidget(self.FactorDBTree, 0, 0, 7, 1)
        self.UpdateButton = QtWidgets.QPushButton(FactorDBDlg)
        self.UpdateButton.setObjectName("UpdateButton")
        self.gridLayout.addWidget(self.UpdateButton, 0, 1, 1, 1)
        self.DescriptionButton = QtWidgets.QPushButton(FactorDBDlg)
        self.DescriptionButton.setObjectName("DescriptionButton")
        self.gridLayout.addWidget(self.DescriptionButton, 1, 1, 1, 1)
        self.ViewButton = QtWidgets.QPushButton(FactorDBDlg)
        self.ViewButton.setObjectName("ViewButton")
        self.gridLayout.addWidget(self.ViewButton, 2, 1, 1, 1)
        self.ScrutinizeButton = QtWidgets.QPushButton(FactorDBDlg)
        self.ScrutinizeButton.setObjectName("ScrutinizeButton")
        self.gridLayout.addWidget(self.ScrutinizeButton, 3, 1, 1, 1)
        self.AdjustGroupBox = QtWidgets.QGroupBox(FactorDBDlg)
        self.AdjustGroupBox.setObjectName("AdjustGroupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.AdjustGroupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.RenameButton = QtWidgets.QPushButton(self.AdjustGroupBox)
        self.RenameButton.setObjectName("RenameButton")
        self.verticalLayout.addWidget(self.RenameButton)
        self.DeleteButton = QtWidgets.QPushButton(self.AdjustGroupBox)
        self.DeleteButton.setObjectName("DeleteButton")
        self.verticalLayout.addWidget(self.DeleteButton)
        self.MoveButton = QtWidgets.QPushButton(self.AdjustGroupBox)
        self.MoveButton.setObjectName("MoveButton")
        self.verticalLayout.addWidget(self.MoveButton)
        self.gridLayout.addWidget(self.AdjustGroupBox, 4, 1, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(FactorDBDlg)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.CSVExportButton = QtWidgets.QPushButton(self.groupBox_2)
        self.CSVExportButton.setObjectName("CSVExportButton")
        self.verticalLayout_2.addWidget(self.CSVExportButton)
        self.CSVImportButton = QtWidgets.QPushButton(self.groupBox_2)
        self.CSVImportButton.setObjectName("CSVImportButton")
        self.verticalLayout_2.addWidget(self.CSVImportButton)
        self.gridLayout.addWidget(self.groupBox_2, 5, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 222, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 6, 1, 1, 1)

        self.retranslateUi(FactorDBDlg)
        QtCore.QMetaObject.connectSlotsByName(FactorDBDlg)

    def retranslateUi(self, FactorDBDlg):
        _translate = QtCore.QCoreApplication.translate
        FactorDBDlg.setWindowTitle(_translate("FactorDBDlg", "因子数据库"))
        self.FactorDBTree.headerItem().setText(0, _translate("FactorDBDlg", "因子"))
        self.FactorDBTree.headerItem().setText(1, _translate("FactorDBDlg", "数据类型"))
        self.UpdateButton.setText(_translate("FactorDBDlg", "刷新"))
        self.DescriptionButton.setText(_translate("FactorDBDlg", "描述信息"))
        self.ViewButton.setText(_translate("FactorDBDlg", "预览"))
        self.ScrutinizeButton.setText(_translate("FactorDBDlg", "详查"))
        self.AdjustGroupBox.setTitle(_translate("FactorDBDlg", "修改调整"))
        self.RenameButton.setText(_translate("FactorDBDlg", "重命名"))
        self.DeleteButton.setText(_translate("FactorDBDlg", "删除"))
        self.MoveButton.setText(_translate("FactorDBDlg", "移动"))
        self.groupBox_2.setTitle(_translate("FactorDBDlg", "导入导出"))
        self.CSVExportButton.setText(_translate("FactorDBDlg", "CSV导出"))
        self.CSVImportButton.setText(_translate("FactorDBDlg", "CSV导入"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    FactorDBDlg = QtWidgets.QDialog()
    ui = Ui_FactorDBDlg()
    ui.setupUi(FactorDBDlg)
    FactorDBDlg.show()
    sys.exit(app.exec_())

