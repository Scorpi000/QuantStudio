# -*- coding: utf-8 -*-
import os
import datetime as dt

import numpy as np
import pandas as pd
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QDialog, QMessageBox, QTableWidgetItem, QFileDialog

from QuantStudio.Tools.QtGUI.Ui_PreviewFactorDlg import Ui_PreviewDlg
from QuantStudio.Tools.QtGUI.DateTimeSetup import DateTimeSetupDlg
from QuantStudio.Tools.QtGUI.IDSetup import IDSetupDlg

class PreviewDlg(QDialog, Ui_PreviewDlg):
    def __init__(self, factor, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.Factor = factor
        #self.DTs = self.Factor.getDateTime(start_dt=dt.datetime.combine(dt.date.today()-dt.timedelta(31), dt.time(0)), end_dt=dt.datetime.today())
        self.DTs = self.Factor.getDateTime()
        self.DTs = self.DTs[-min(60, len(self.DTs)):]
        self.IDs = self.Factor.getID()
        self.IDs = self.IDs[:min(10, len(self.IDs))]
        self.FactorData = self.Factor.readData(ids=self.IDs, dts=self.DTs)
        self.DimChanged = False
        self.setWindowTitle("预览: "+self.Factor.Name)
        self.populatePreviewTable()
        self.populateDateTime()
        self.populateID()
        return
    def populatePreviewTable(self):
        self.PreviewTable.clear()
        nRow, nCol = self.FactorData.shape
        self.PreviewTable.setColumnCount(nCol)
        self.PreviewTable.setHorizontalHeaderLabels(self.FactorData.columns.tolist())
        self.PreviewTable.setRowCount(nRow)
        if nRow==0: return 0
        if ((self.FactorData.shape[0]>1) and (int(np.diff(self.FactorData.index.values).min()/10**9/3600/24)<1)) or ((self.FactorData.shape[0]==1) and (self.FactorData.index[0].to_pydatetime().time()!=dt.time(0))):
            self.PreviewTable.setVerticalHeaderLabels([iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in self.FactorData.index])
        else:
            self.PreviewTable.setVerticalHeaderLabels([iDT.strftime("%Y-%m-%d") for iDT in self.FactorData.index])
        for i in range(nRow):
            for j in range(nCol):
                self.PreviewTable.setItem(i, j, QTableWidgetItem(str(self.FactorData.iloc[i, j])))
        return 0
    @pyqtSlot()
    def on_ExecuteButton_clicked(self):
        self.ExecuteButton.setText("加载中...")
        self.ExecuteButton.setEnabled(False)
        if self.DimChanged: 
            self.FactorData = self.Factor.readData(ids=self.IDs, dts=self.DTs)
            self.DimChanged = False
        self.populatePreviewTable()
        self.ExecuteButton.setText("预览")
        self.ExecuteButton.setEnabled(True)
        return 0
    def populateDateTime(self):
        self.DateTimeList.clear()
        self.DateTimeList.addItems([iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in self.DTs])
        return 0
    @pyqtSlot()
    def on_DateTimeButton_clicked(self):
        Dlg = DateTimeSetupDlg(self, dts=self.DTs, ft=self.Factor.FactorTable)
        Dlg.exec_()
        if Dlg.isChanged:
            self.DTs = Dlg.DateTimes
            self.DimChanged = True
        return self.populateDateTime()
    def populateID(self):
        self.IDList.clear()
        self.IDList.addItems(self.IDs)
        return 0
    @pyqtSlot()
    def on_IDButton_clicked(self):
        Dlg = IDSetupDlg(self, ids=self.IDs, ft=self.Factor.FactorTable)
        Dlg.exec_()
        if Dlg.isChanged:
            self.IDs = Dlg.IDs
            self.DimChanged = True
        return self.populateID()
    @pyqtSlot()
    def on_Export2CSVButton_clicked(self):
        FilePath, _ = QFileDialog.getSaveFileName(self, "导出数据", os.getcwd()+os.sep+"untitled.csv", "csv (*.csv)")
        if not FilePath: return 0
        if self.DimChanged:
            self.FactorData = self.Factor.readData(ids=self.IDs, dts=self.DTs)
            self.DimChanged = False
        self.FactorData.to_csv(FilePath)
        return QMessageBox.information(self, "完成", "导出数据完成!")
    @pyqtSlot()
    def on_ArgSetButton_clicked(self):
        self.DimChanged = self.Factor.setArgs()
        return 0

if __name__=="__main__":
    import QuantStudio.api as QS
    FDB = QS.FactorDB.WindDB2()
    FDB.connect()
    FT = FDB.getTable("中国A股日行情")
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    Factor = FT.getFactor("收盘价(元)")
    TestWindow = PreviewDlg(factor=Factor)
    TestWindow.show()
    app.exec_()
    sys.exit()