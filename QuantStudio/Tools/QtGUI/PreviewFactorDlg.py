# -*- coding: utf-8 -*-
import os
import datetime as dt

import numpy as np
import pandas as pd
from PyQt5.QtCore import pyqtSlot, QModelIndex
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
        self.AllDTs = self.Factor.getDateTime()
        self.DTs = self.AllDTs[-min(60, len(self.AllDTs)):]
        self.AllDTs = pd.Series(np.arange(0, len(self.AllDTs)), index=self.AllDTs)
        self.AllIDs = self.Factor.getID()
        self.IDs = self.AllIDs[:min(10, len(self.AllIDs))]
        self.AllIDs = pd.Series(np.arange(0, len(self.AllIDs)), index=self.AllIDs)
        self.FactorData = self.Factor.readData(ids=self.IDs, dts=self.DTs)
        self.setWindowTitle("预览: "+self.Factor.Name)
        self.populatePreviewTable()
        self.IDList.addItems(self.AllIDs.index.tolist())
        self.DateTimeList.addItems([str(iDT) for iDT in self.AllDTs.index])
        self.updateSelectedDateTime()
        self.updateSelectedID()
        self.DateTimeList.scrollToBottom()
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
        self.FactorData = self.Factor.readData(ids=self.IDs, dts=self.DTs)
        self.populatePreviewTable()
        self.ExecuteButton.setText("刷新")
        self.ExecuteButton.setEnabled(True)
        return 0
    def updateSelectedDateTime(self):
        for i in self.AllDTs[self.DTs]:
            self.DateTimeList.item(i).setSelected(True)
        if self.DTs:
            self.DateTimeList.scrollToItem(self.DateTimeList.item(self.AllDTs[self.DTs[0]]))
        return 0
    @pyqtSlot()
    def on_DateTimeButton_clicked(self):
        Dlg = DateTimeSetupDlg(self, dts=self.AllDTs, ft=self.Factor.FactorTable)
        Dlg.exec_()
        if Dlg.isChanged:
            self.DTs = sorted(self.AllDTs.index.intersection(Dlg.DateTimes))
            self.updateSelectedDateTime()
            self.on_DTExecuteButton_clicked()
        return 0
    def updateSelectedID(self):
        for i in self.AllIDs[self.IDs]:
            self.IDList.item(i).setSelected(True)
        if self.IDs:
            self.IDList.scrollToItem(self.IDList.item(self.AllIDs[self.IDs[0]]))
        return 0
    @pyqtSlot()
    def on_IDButton_clicked(self):
        Dlg = IDSetupDlg(self, ids=self.AllIDs.index.tolist(), ft=self.Factor.FactorTable)
        Dlg.exec_()
        if Dlg.isChanged:
            self.IDs = sorted(self.AllIDs.index.intersection(Dlg.IDs))
            self.updateSelectedID()
            self.on_IDExecuteButton_clicked()
        return 0
    @pyqtSlot()
    def on_Export2CSVButton_clicked(self):
        FilePath, _ = QFileDialog.getSaveFileName(self, "导出数据", os.getcwd()+os.sep+"untitled.csv", "csv (*.csv)")
        if not FilePath: return 0
        self.FactorData.to_csv(FilePath)
        return QMessageBox.information(self, "完成", "导出数据完成!")
    @pyqtSlot()
    def on_ArgSetButton_clicked(self):
        self.Factor.setArgs()
        return 0
    @pyqtSlot()
    def on_DTExecuteButton_clicked(self):
        SelectedIndexes = self.DateTimeList.selectedIndexes()
        self.DTs = [self.AllDTs.index[iIdx.row()] for iIdx in SelectedIndexes]
        self.FactorData = self.Factor.readData(ids=self.IDs, dts=self.DTs)
        self.populatePreviewTable()
        return 0
    @pyqtSlot(QModelIndex)
    def on_DateTimeList_doubleClicked(self, index):
        return self.on_DTExecuteButton_clicked()
    @pyqtSlot()
    def on_IDExecuteButton_clicked(self):
        SelectedIndexes = self.IDList.selectedIndexes()
        self.IDs = [self.AllIDs.index[iIdx.row()] for iIdx in SelectedIndexes]
        self.FactorData = self.Factor.readData(ids=self.IDs, dts=self.DTs)
        self.populatePreviewTable()
        return 0
    @pyqtSlot(QModelIndex)
    def on_IDList_doubleClicked(self, index):
        return self.on_IDExecuteButton_clicked()

if __name__=="__main__":
    import QuantStudio.api as QS
    FDB = QS.FactorDB.HDF5DB()
    FDB.connect()
    FT = FDB.getTable("ElementaryFactor")
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    Factor = FT.getFactor("复权收盘价")
    TestWindow = PreviewDlg(factor=Factor)
    TestWindow.show()
    app.exec_()
    sys.exit()