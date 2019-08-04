# -*- coding: utf-8 -*-
import os
import datetime as dt

import numpy as np
import pandas as pd
from PyQt5.QtCore import pyqtSlot, QModelIndex
from PyQt5.QtWidgets import QDialog, QTreeWidgetItem, QMessageBox, QInputDialog, QFileDialog

from QuantStudio.Tools.QtGUI.Ui_RiskDBDlg import Ui_RiskDBDlg
from QuantStudio.Tools.QtGUI.PreviewRiskDataDlg import PreviewRiskDataDlg
from QuantStudio.Tools.AuxiliaryFun import genAvailableName

class RiskDBDlg(QDialog, Ui_RiskDBDlg):
    def __init__(self, risk_db, parent=None):
        super(RiskDBDlg, self).__init__(parent)
        self.setupUi(self)
        self.RiskDB = risk_db
        self.populateRiskDBTree()
    def isBottomItemSelected(self):
        SelectedItem = self.RiskDBTree.selectedItems()
        if (len(SelectedItem)==0) or (SelectedItem[0].parent() is None):
            return (False, "选择的不是一个时点!")
        else:
            return (True, SelectedItem[0])
    def isTopItemSelected(self):
        SelectedItem = self.RiskDBTree.selectedItems()
        if (len(SelectedItem)==0) or (SelectedItem[0].parent() is not None):
            return (False, "选择的不是一张风险表!")
        else:
            return (True, SelectedItem[0])
    def genTableDTs(self):# 产生当前 RiskDBTree 中选择的 TableDTs: {表名：[时点]}
        SelectedItems = self.RiskDBTree.selectedItems()
        TableDTs = {}
        for iItem in SelectedItems:
            if iItem.parent() is None:# 该项是表
                iTable = iItem.text(0)
                TableDTs[iTable] = None
            else:# 该项是因子
                iTable = iItem.parent().text(0)
                if iTable in TableDTs:
                    iDT = iItem.text(0)
                    if TableDTs[iTable] is not None:
                        TableDTs[iTable].append(dt.datetime.strptime(iDT, "%Y-%m-%d %H:%M:%S.%f"))
                else:
                    TableDTs[iTable] = [iItem.text(0)]
        return TableDTs
    def getNewTableName(self, init_table_name=None):# 获取新表名
        if init_table_name is None:
            TableName = genAvailableName(header="NewTable", all_names=self.RiskDB.TableNames)
        elif init_table_name in self.RiskDB.TableNames:
            TableName = genAvailableName(header=init_table_name, all_names=self.RiskDB.TableNames)
        else:
            TableName = init_table_name
        TableName, isOk = QInputDialog.getText(self, "新表名", "请输入新表名", text=TableName)
        if not isOk: return (False, "")
        if TableName in self.RiskDB.TableNames:
            QMessageBox.critical(self, "错误", "当前包含重名表!")
            return (False, "")
        return (True, TableName)
    def getNewDateTime(self):# 获取新时点
        NewDT, isOk = QInputDialog.getText(self, "新时点", "请输入新时点", text=dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S.%f"))
        if not isOk: return (False, "")
        try:
            NewDT = dt.datetime.strptime(NewDT, "%Y-%m-%d %H:%M:%S.%f")
        except Exception as e:
            QMessageBox.critical(self, "错误", "时点格式错误!")
            return (False, "")
        return (True, NewDT)
    def populateRiskDBTree(self):
        self.RiskDBTree.blockSignals(True)
        self.RiskDBTree.clear()
        self.RiskDBTree.setHeaderLabels(["时点"])
        for iTableName in self.RiskDB.TableNames:
            Parent = QTreeWidgetItem(self.RiskDBTree, [iTableName])
            QTreeWidgetItem(Parent, ["_时点"])
        self.RiskDBTree.resizeColumnToContents(0)
        self.RiskDBTree.blockSignals(False)
        return 0
    @pyqtSlot(QModelIndex)
    def on_RiskDBTree_expanded(self, index):
        Item = self.RiskDBTree.itemFromIndex(index)
        FactorItem = Item.child(0)
        if FactorItem.text(0)=="_时点":
            Item.removeChild(FactorItem)
            TableName = Item.text(0)
            RT = self.RiskDB.getTable(TableName)
            for i, iDateTime in enumerate(RT.getDateTime()):
                QTreeWidgetItem(Item, [iDateTime.strftime("%Y-%m-%d %H:%M:%S.%f")])
    @pyqtSlot()
    def on_UpdateButton_clicked(self):
        try:
            self.RiskDB.connect()
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            return 0
        return self.populateRiskDBTree()
    @pyqtSlot()
    def on_DescriptionButton_clicked(self):
        isDTSelected, SelectedItem = self.isBottomItemSelected()
        if isDTSelected:
            SelectedTableName = SelectedItem.parent().text(0)
        else:
            isTableSelected, SelectedItem = self.isTopItemSelected()
            if isTableSelected:
                SelectedTableName = SelectedItem.text(0)
            else:
                return QMessageBox.critical(self, "错误", SelectedItem)
        Description = self.RiskDB.getTable(SelectedTableName).getMetaData(key="Description")
        return QMessageBox.information(self, "描述信息", str(Description))
    @pyqtSlot()
    def on_ViewButton_clicked(self):
        # 检查用户是否选择了一个时点
        isDTSelected, SelectedItem = self.isBottomItemSelected()
        if not isDTSelected:
            return QMessageBox.critical(None, "错误", SelectedItem)
        DT = dt.datetime.strptime(SelectedItem.text(0), "%Y-%m-%d %H:%M:%S.%f")
        Dlg = PreviewRiskDataDlg(risk_db=self.RiskDB, table_name=SelectedItem.parent().text(0), idt=DT, parent=None)
        return Dlg.exec_()
    @pyqtSlot()
    def on_RenameButton_clicked(self):
        # 检查用户是否选择了一张表
        isTableSelected, SelectedItem = self.isTopItemSelected()
        if isTableSelected:
            OldTableName = SelectedItem.text(0)
            # 获取新表名
            NewTableName, isOk = QInputDialog.getText(self, "表名", "请输入表名", text=OldTableName)
            if (not isOk) or (OldTableName==NewTableName): return 0
            if NewTableName in self.RiskDB.TableNames:
                return QMessageBox.critical(self, "错误", "当前包含重名表!")
            # 调整其他关联区的数据
            try:
                self.RiskDB.renameTable(OldTableName, NewTableName)
                SelectedItem.setText(0, NewTableName)
            except Exception as e:
                QMessageBox.critical(self, "错误", str(e))
            return 0
        else:
            return QMessageBox.critical(self, "错误", SelectedItem)
    @pyqtSlot()
    def on_DeleteButton_clicked(self):
        isOK = QMessageBox.question(self, "删除", "删除后将无法恢复, 你是否能对自己的行为负责?", QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel)
        if isOK!=QMessageBox.Ok: return 0
        TableDTs = self.genTableDTs()
        for iTable in TableDTs:
            try:
                if TableDTs[iTable] is None:
                    self.RiskDB.deleteTable(iTable)
                else:
                    self.RiskDB.deleteDateTime(iTable, TableDTs[iTable])
            except Exception as e:
                QMessageBox.critical(self, "错误", str(e))
        return self.populateRiskDBTree()
    @pyqtSlot()
    def on_CSVExportButton_clicked(self):
        DirPath = QFileDialog.getExistingDirectory(parent=self, caption="导出CSV", directory=os.getcwd())
        if not DirPath: return 0
        self.setEnabled(False)
        TableDTs = self.genTableDTs()
        for iTable, iDTs in TableDTs.items():
            iRT = self.RiskDB.getTable(iTable)
            if iDTs is None: iDTs = iRT.getDateTime()
            iData = iRT.readCov(dts=iDTs)
            for j, jDT in enumerate(iData.items):
                iData.iloc[j].to_csv(DirPath+os.sep+iTable+"-"+jDT.strftime("%Y-%m-%d %H-%M-%S-%f")+".csv", encoding="utf-8")
        QMessageBox.information(self, "完成", "导出数据完成!")
        self.setEnabled(True)
        return 0
    @pyqtSlot()
    def on_CSVImportButton_clicked(self):
        SelectedItems = self.RiskDBTree.selectedItems()
        nSelectedItems = len(SelectedItems)
        if (nSelectedItems>1):
            return QMessageBox.critical(self, "错误", "请选择一张表或一个时点!")
        elif nSelectedItems==0:
            # 获取新表名
            isOk, TableName = self.getNewTableName()
            if not isOk: return 0
            isOK, NewDT = self.getNewDateTime()
            if not isOk: return 0
        else:
            if SelectedItems[0].parent() is None:
                TableName = SelectedItems[0].text(0)
                isOk, NewDT = self.getNewDateTime()
                if not isOk: return 0
            else:
                TableName = SelectedItems[0].parent().text(0)
                NewDT = dt.datetime.strptime(SelectedItems[0].text(0), "%Y-%m-%d %H:%M:%S.%f")
        FilePath = QFileDialog.getOpenFileName(parent=self, caption="导入CSV", directory=".", filter="csv (*.csv)")[0]
        if not FilePath: return 0
        try:
            RiskData = pd.read_csv(FilePath, header=0, index_col=0, engine="python")
        except Exception as e:
            return QMessageBox.critical(self, "错误", "文件读取失败: "+str(e))
        self.setEnabled(False)
        try:
            self.RiskDB.writeData(TableName, NewDT, RiskData)
        except Exception as e:
            QMessageBox.critical(self, "错误", "数据写入失败: "+str(e))
        else:
            self.populateRiskDBTree()
            QMessageBox.information(self, "完成", "导入数据完成!")
        finally:
            self.setEnabled(True)
        return 0

if __name__=="__main__":
    import QuantStudio.api as QS
    RDB = QS.RiskDB.HDF5FRDB()
    RDB.connect()
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    TestWindow = RiskDBDlg(risk_db=RDB)
    TestWindow.show()
    sys.exit(app.exec_())