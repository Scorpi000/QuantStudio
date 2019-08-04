# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Project\QuantStudio\QuantStudio\Tools\QtGUI\RiskDBDlg.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_RiskDBDlg(object):
    def setupUi(self, RiskDBDlg):
        RiskDBDlg.setObjectName("RiskDBDlg")
        RiskDBDlg.resize(412, 539)
        RiskDBDlg.setSizeGripEnabled(True)
        self.gridLayout = QtWidgets.QGridLayout(RiskDBDlg)
        self.gridLayout.setObjectName("gridLayout")
        self.RiskDBTree = QtWidgets.QTreeWidget(RiskDBDlg)
        self.RiskDBTree.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.RiskDBTree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.RiskDBTree.setObjectName("RiskDBTree")
        self.gridLayout.addWidget(self.RiskDBTree, 0, 0, 6, 1)
        self.UpdateButton = QtWidgets.QPushButton(RiskDBDlg)
        self.UpdateButton.setObjectName("UpdateButton")
        self.gridLayout.addWidget(self.UpdateButton, 0, 1, 1, 1)
        self.DescriptionButton = QtWidgets.QPushButton(RiskDBDlg)
        self.DescriptionButton.setObjectName("DescriptionButton")
        self.gridLayout.addWidget(self.DescriptionButton, 1, 1, 1, 1)
        self.ViewButton = QtWidgets.QPushButton(RiskDBDlg)
        self.ViewButton.setObjectName("ViewButton")
        self.gridLayout.addWidget(self.ViewButton, 2, 1, 1, 1)
        self.AdjustGroupBox = QtWidgets.QGroupBox(RiskDBDlg)
        self.AdjustGroupBox.setObjectName("AdjustGroupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.AdjustGroupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.RenameButton = QtWidgets.QPushButton(self.AdjustGroupBox)
        self.RenameButton.setObjectName("RenameButton")
        self.verticalLayout.addWidget(self.RenameButton)
        self.DeleteButton = QtWidgets.QPushButton(self.AdjustGroupBox)
        self.DeleteButton.setObjectName("DeleteButton")
        self.verticalLayout.addWidget(self.DeleteButton)
        self.gridLayout.addWidget(self.AdjustGroupBox, 3, 1, 1, 1)
        self.ImExportBox = QtWidgets.QGroupBox(RiskDBDlg)
        self.ImExportBox.setObjectName("ImExportBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.ImExportBox)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.CSVExportButton = QtWidgets.QPushButton(self.ImExportBox)
        self.CSVExportButton.setObjectName("CSVExportButton")
        self.verticalLayout_2.addWidget(self.CSVExportButton)
        self.CSVImportButton = QtWidgets.QPushButton(self.ImExportBox)
        self.CSVImportButton.setObjectName("CSVImportButton")
        self.verticalLayout_2.addWidget(self.CSVImportButton)
        self.gridLayout.addWidget(self.ImExportBox, 4, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 222, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 5, 1, 1, 1)

        self.retranslateUi(RiskDBDlg)
        QtCore.QMetaObject.connectSlotsByName(RiskDBDlg)

    def retranslateUi(self, RiskDBDlg):
        _translate = QtCore.QCoreApplication.translate
        RiskDBDlg.setWindowTitle(_translate("RiskDBDlg", "风险数据库"))
        self.RiskDBTree.headerItem().setText(0, _translate("RiskDBDlg", "时点"))
        self.UpdateButton.setText(_translate("RiskDBDlg", "刷新"))
        self.DescriptionButton.setText(_translate("RiskDBDlg", "描述信息"))
        self.ViewButton.setText(_translate("RiskDBDlg", "预览"))
        self.AdjustGroupBox.setTitle(_translate("RiskDBDlg", "修改调整"))
        self.RenameButton.setText(_translate("RiskDBDlg", "重命名"))
        self.DeleteButton.setText(_translate("RiskDBDlg", "删除"))
        self.ImExportBox.setTitle(_translate("RiskDBDlg", "导入导出"))
        self.CSVExportButton.setText(_translate("RiskDBDlg", "CSV导出"))
        self.CSVImportButton.setText(_translate("RiskDBDlg", "CSV导入"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    RiskDBDlg = QtWidgets.QDialog()
    ui = Ui_RiskDBDlg()
    ui.setupUi(RiskDBDlg)
    RiskDBDlg.show()
    sys.exit(app.exec_())

