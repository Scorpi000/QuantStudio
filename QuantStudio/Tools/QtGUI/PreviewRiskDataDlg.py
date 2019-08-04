# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QDialog, QMessageBox, QTableWidgetItem, QFileDialog, QAction

from QuantStudio.Tools.QtGUI.Ui_PreviewRiskDataDlg import Ui_Dialog
from QuantStudio.RiskDataBase.RiskDB import FactorRT


class PreviewRiskDataDlg(QDialog, Ui_Dialog):
    def __init__(self, risk_db, table_name, idt, parent=None):
        super(PreviewRiskDataDlg, self).__init__(parent)
        self.setupUi(self)
        self.RiskDB = risk_db
        self.RT = self.RiskDB.getTable(table_name)
        self.DT = idt
        self.RiskData = self.RT.readCov(dts=[self.DT], ids=None).iloc[0]
        self.IDs = self.RiskData.index.tolist()
        self.setWindowTitle("预览: "+self.DT.strftime("%Y-%m-%d %H:%M:%S.%f"))
        self.PreviewTable.addAction(QAction("去除缺失", self.PreviewTable, triggered=self.dropNA))
        self.populatePreviewTable()
        self.populateID()
        if isinstance(self.RT, FactorRT):
            self.populateFactor()
        else:
            self.PreviewFactorRiskButton.setEnabled(False)
            self.PreviewSpecificRiskButton.setEnabled(False)
    def populatePreviewTable(self):
        self.PreviewTable.clear()
        nRow, nCol = self.RiskData.shape
        if self.RowNumLimitCheckBox.isChecked(): nRow = min(nRow, self.RowNumSpinBox.value())
        if self.ColNumLimitCheckBox.isChecked(): nCol = min(nCol, self.ColNumSpinBox.value())
        self.PreviewTable.setColumnCount(nCol)
        self.PreviewTable.setHorizontalHeaderLabels(self.RiskData.columns[:nCol].tolist())
        self.PreviewTable.setRowCount(nRow)
        self.PreviewTable.setVerticalHeaderLabels(self.RiskData.index[:nRow].tolist())
        for i in range(nRow):
            for j in range(nCol):
                self.PreviewTable.setItem(i, j, QTableWidgetItem(str(self.RiskData.iloc[i, j])))
        return 0
    def populateFactor(self):
        FactorNames = self.RT.FactorNames
        self.FactorList.clear()
        self.FactorList.addItems(FactorNames)
        self.FactorListLabel.setText("因子: %d" % len(FactorNames))
        return 0    
    def populateID(self):
        self.IDList.clear()
        self.IDList.addItems(self.IDs)
        self.IDListLabel.setText("ID: %d" % len(self.IDs))
        return 0
    def dropNA(self):
        self.RiskData = self.RiskData.dropna(how="all", axis=0)
        if self.RiskData.shape[1]>1:
            self.RiskData = self.RiskData.loc[:, self.RiskData.index]
            self.RiskData = self.RiskData.dropna(how="any", axis=0)
            self.RiskData = self.RiskData.loc[:, self.RiskData.index]
        return self.populatePreviewTable()
    @pyqtSlot()
    def on_PreviewFactorRiskButton_clicked(self):
        self.RiskData = self.RT.readFactorCov(dts=[self.DT]).iloc[0]
        return self.populatePreviewTable()
    @pyqtSlot()
    def on_PreviewSpecificRiskButton_clicked(self):
        self.RiskData = self.RT.readSpecificRisk(dts=[self.DT], ids=None).iloc[0]
        self.RiskData = pd.DataFrame(self.RiskData)
        self.RiskData.columns = ["SpecificRisk"]
        return self.populatePreviewTable()
    @pyqtSlot()
    def on_PreviewRiskButton_clicked(self):
        self.RiskData = self.RT.readCov(dts=[self.DT], ids=None).iloc[0]
        return self.populatePreviewTable()
    @pyqtSlot()
    def on_Export2CSVButton_clicked(self):
        FilePath, _ = QFileDialog.getSaveFileName(self, "导出数据", os.getcwd()+os.sep+"RiskData.csv", "csv (*.csv)")
        if not FilePath: return 0
        self.RiskData.to_csv(FilePath)
        return QMessageBox.information(self, "完成", "导出数据完成!")