# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from QuantStudio.GUI.Ui_IDFilter import Ui_IDFilterDlg, _fromUtf8

from QuantStudio.FunLib.IDFun import testIDFilterStr
from QuantStudio.GUI.QtGUIObject import HDF5LibTreeWidget


class IDFilterDlg(QDialog, Ui_IDFilterDlg):
    def __init__(self, parent=None, factor_names=[], id_filter_str=None, qs_env=None):
        super().__init__(parent)
        self.setupUi(self)
        self.IDFilterStr = id_filter_str
        self.QSEnv = qs_env
        self.LibPath = self.QSEnv.SysArgs['LibPath']+os.sep+'IDFilter.hdf5'
        self.FactorNames = factor_names
        self.TestDF = pd.DataFrame(columns=self.FactorNames)
        # 设置 IDFilterLibTree
        self.IDFilterLibTree = HDF5LibTreeWidget(self, self.LibPath, load_item_fun=self.loadLibFilter, 
                                                 save_item_fun=self.saveLibFilter, qs_env=self.QSEnv)
        self.IDFilterLibTree.setObjectName(_fromUtf8("IDFilterLibTree"))
        self.gridLayout_3.addWidget(self.IDFilterLibTree, 1, 1, 3, 1)
        self.isChanged = False
        # 填充 DSList
        self.FactorNames.sort()
        self.DSList.addItems(self.FactorNames)
        # 设置ConditionTextEdit
        if self.IDFilterStr is not None:
            self.ConditionTextEdit.setText(self.IDFilterStr)
        return
    def loadLibFilter(self, lib_filter):
        if isinstance(lib_filter, dict):
            QMessageBox.critical(self, "错误", "请选择一个过滤条件!")
            return 0
        self.ConditionTextEdit.clear()
        self.ConditionTextEdit.setText(lib_filter)
        return 0
    def saveLibFilter(self):
        ConditionStr = self.ConditionTextEdit.toPlainText()
        ConditionStr = ConditionStr.replace('\n','')
        return ConditionStr
    @pyqtSlot()
    def on_TestButton_clicked(self):
        ConditionStr = self.ConditionTextEdit.toPlainText()
        ConditionStr = ConditionStr.replace('\n','')
        CompiledStr, FactorList = testIDFilterStr(ConditionStr, self.FactorNames.copy())
        if CompiledStr is not None:
            QMessageBox.information(None,'成功',"条件字符串测试通过!")
        else:
            QMessageBox.critical(None,'失败','输入的条件字符串有误!')
        return 0
    @pyqtSlot()
    def on_FinishButtonBox_accepted(self):
        ConditionStr = self.ConditionTextEdit.toPlainText()
        ConditionStr = ConditionStr.replace('\n','')
        if ConditionStr!="":
            CompiledStr, FactorList = testIDFilterStr(ConditionStr, self.FactorNames.copy())
            if CompiledStr is None:
                QMessageBox.critical(None,'失败','输入的条件字符串有误!')
                return 0
            self.IDFilterStr = ConditionStr
        else:
            self.IDFilterStr = None
        self.isChanged = True
        self.close()
        return 0
    @pyqtSlot()
    def on_FinishButtonBox_rejected(self):
        self.close()
        return 0
    @pyqtSlot(QListWidgetItem)
    def on_DSList_itemDoubleClicked(self, item):
        FactorName = item.text()
        self.ConditionTextEdit.insertPlainText('@'+FactorName)
        return 0
    @pyqtSlot()
    def on_PlusButton_clicked(self):
        self.ConditionTextEdit.insertPlainText('+')
        return 0
    @pyqtSlot()
    def on_MinusButton_clicked(self):
        self.ConditionTextEdit.insertPlainText('-')
        return 0
    @pyqtSlot()
    def on_TimeButton_clicked(self):
        self.ConditionTextEdit.insertPlainText('*')
        return 0
    @pyqtSlot()
    def on_DivideButton_clicked(self):
        self.ConditionTextEdit.insertPlainText('/')
        return 0
    @pyqtSlot()
    def on_PowerButton_clicked(self):
        self.ConditionTextEdit.insertPlainText('**')
        return 0
    @pyqtSlot()
    def on_GreaterButton_clicked(self):
        self.ConditionTextEdit.insertPlainText('>')
        return 0
    @pyqtSlot()
    def on_GreaterEqualButton_clicked(self):
        self.ConditionTextEdit.insertPlainText('>=')
        return 0
    @pyqtSlot()
    def on_EqualButton_clicked(self):
        self.ConditionTextEdit.insertPlainText('==')
        return 0
    @pyqtSlot()
    def on_NotEqualButton_clicked(self):
        self.ConditionTextEdit.insertPlainText('!=')
        return 0
    @pyqtSlot()
    def on_LessEqualButton_clicked(self):
        self.ConditionTextEdit.insertPlainText('<=')
        return 0
    @pyqtSlot()
    def on_LessButton_clicked(self):
        self.ConditionTextEdit.insertPlainText('<')
        return 0
    @pyqtSlot()
    def on_AndButton_clicked(self):
        self.ConditionTextEdit.insertPlainText(' & ')
        return 0
    @pyqtSlot()
    def on_OrButton_clicked(self):
        self.ConditionTextEdit.insertPlainText(' | ')
        return 0
    @pyqtSlot()
    def on_NotButton_clicked(self):
        self.ConditionTextEdit.insertPlainText('~ ')
        return 0
    @pyqtSlot()
    def on_NotNullButton_clicked(self):
        TextCuror = self.ConditionTextEdit.textCursor()
        SelectedStr = TextCuror.selectedText()
        if SelectedStr=='':
            self.ConditionTextEdit.insertPlainText('pd.notnull(')
        else:
            CurStr = self.ConditionTextEdit.toPlainText()
            StartPos = TextCuror.selectionStart()
            EndPos = TextCuror.selectionEnd()
            self.ConditionTextEdit.setText(CurStr[:StartPos]+'pd.notnull('+SelectedStr+')'+CurStr[EndPos:])
        return 0
    @pyqtSlot()
    def on_NullButton_clicked(self):
        TextCuror = self.ConditionTextEdit.textCursor()
        SelectedStr = TextCuror.selectedText()
        if SelectedStr=='':
            self.ConditionTextEdit.insertPlainText('pd.isnull(')
        else:
            CurStr = self.ConditionTextEdit.toPlainText()
            StartPos = TextCuror.selectionStart()
            EndPos = TextCuror.selectionEnd()
            self.ConditionTextEdit.setText(CurStr[:StartPos]+'pd.isnull('+SelectedStr+')'+CurStr[EndPos:])
        return 0
    @pyqtSlot()
    def on_BracketsButton_clicked(self):
        TextCuror = self.ConditionTextEdit.textCursor()
        SelectedStr = TextCuror.selectedText()
        CurStr = self.ConditionTextEdit.toPlainText()
        if SelectedStr=='':
            if CurStr!='':
                self.ConditionTextEdit.setText('('+CurStr+')')
            else:
                self.ConditionTextEdit.setText('()')
        else:
            StartPos = TextCuror.selectionStart()
            EndPos = TextCuror.selectionEnd()
            self.ConditionTextEdit.setText(CurStr[:StartPos]+'('+SelectedStr+')'+CurStr[EndPos:])
        return 0

if __name__=='__main__':
    import os
    import sys
    app = QApplication(sys.argv)
    from QuantStudio import QSEnv
    QSE = QSEnv()
    Dlg = IDFilterDlg(None, ["a","b","c"], "@a==1", QSE)
    Dlg.show()
    app.exec_()
    sys.exit()