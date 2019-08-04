# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
from PyQt5.QtCore import pyqtSlot, QModelIndex
from PyQt5.QtWidgets import QDialog, QTreeWidgetItem, QMessageBox, QInputDialog, QFileDialog
from QuantStudio.Tools.QtGUI.Ui_FactorDBDlg import Ui_FactorDBDlg

from QuantStudio.Tools.QtGUI.PreviewFactorDlg import PreviewDlg
from QuantStudio.Tools.FileFun import loadCSVFactorData
from QuantStudio.Tools.AuxiliaryFun import genAvailableName
from QuantStudio.FactorDataBase.FactorDB import WritableFactorDB

class FactorDBDlg(QDialog, Ui_FactorDBDlg):
    def __init__(self, factor_db, parent=None):
        super(FactorDBDlg, self).__init__(parent)
        self.setupUi(self)
        self.FactorDB = factor_db
        self.populateFactorDBTree()
        if not isinstance(self.FactorDB, WritableFactorDB):
            self.AdjustGroupBox.setEnabled(False)
            self.CSVImportButton.setEnabled(False)
    def populateFactorDBTree(self):
        self.FactorDBTree.blockSignals(True)
        self.FactorDBTree.clear()
        self.FactorDBTree.setHeaderLabels(["因子", "数据类型"])
        for iTableName in self.FactorDB.TableNames:
            Parent = QTreeWidgetItem(self.FactorDBTree, [iTableName])
            QTreeWidgetItem(Parent, ["_因子", "_数据类型"])
        self.FactorDBTree.resizeColumnToContents(0)
        self.FactorDBTree.blockSignals(False)
        return 0
    def isBottomItemSelected(self):
        SelectedItem = self.FactorDBTree.selectedItems()
        if (len(SelectedItem)==0) or (SelectedItem[0].parent() is None):
            return (False, "选择的不是一个因子!")
        else:
            return (True, SelectedItem[0])
    def isTopItemSelected(self):
        SelectedItem = self.FactorDBTree.selectedItems()
        if (len(SelectedItem)==0) or (SelectedItem[0].parent() is not None):
            return (False, "选择的不是一张因子表!")
        else:
            return (True, SelectedItem[0])
    def genTableFactor(self):# 产生当前 FactorDBTree 中选择的 TableFactor: {表名：[因子名]}
        SelectedItems = self.FactorDBTree.selectedItems()
        TableFactor = {}
        for iItem in SelectedItems:
            if iItem.parent() is None:# 该项是表
                iTable = iItem.text(0)
                TableFactor[iTable] = None
            else:# 该项是因子
                iTable = iItem.parent().text(0)
                if iTable in TableFactor:
                    iFactor = iItem.text(0)
                    if TableFactor[iTable] is not None:
                        TableFactor[iTable].append(iFactor)
                else:
                    TableFactor[iTable] = [iItem.text(0)]
        return TableFactor
    def getNewTableName(self, init_table_name=None):# 获取新表名
        if init_table_name is None:
            TableName = genAvailableName(header="NewTable", all_names=self.FactorDB.TableNames)
        elif init_table_name in self.FactorDB.TableNames:
            TableName = genAvailableName(header=init_table_name, all_names=self.FactorDB.TableNames)
        else:
            TableName = init_table_name
        TableName, isOk = QInputDialog.getText(self, "新表名", "请输入新表名", text=TableName)
        if not isOk: return (False, "")
        if TableName in self.FactorDB.TableNames:
            QMessageBox.critical(self, "错误", "当前包含重名表!")
            return (False, "")
        return (True, TableName)
    def getNewFactorName(self, table_name):# 获取新因子名
        AllFactorNames = self.FactorDB.getTable(table_name).FactorNames
        NewFactorName = genAvailableName(header="NewFactor", all_names=AllFactorNames)
        NewFactorName, isOk = QInputDialog.getText(self, "新因子名", "请输入新因子名", text=NewFactorName)
        if not isOk: return (False, "")
        if NewFactorName in AllFactorNames:
            QMessageBox.critical(self, "错误", "当前表包含重名因子!")
            return (False, "")
        return (True, NewFactorName)
    @pyqtSlot(QModelIndex)
    def on_FactorDBTree_expanded(self, index):
        Item = self.FactorDBTree.itemFromIndex(index)
        FactorItem = Item.child(0)
        if FactorItem.text(0)=="_因子":
            Item.removeChild(FactorItem)
            TableName = Item.text(0)
            FT = self.FactorDB.getTable(TableName)
            DataType = FT.getFactorMetaData(FT.FactorNames, key="DataType")
            for i, iFactorName in enumerate(DataType.index):
                QTreeWidgetItem(Item, [iFactorName, DataType.iloc[i]])
    @pyqtSlot()
    def on_UpdateButton_clicked(self):
        try:
            self.FactorDB.connect()
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            return 0
        return self.populateFactorDBTree()
    @pyqtSlot()
    def on_DescriptionButton_clicked(self):
        isTableSelected, SelectedItem = self.isTopItemSelected()
        if isTableSelected:
            SelectedTableName = SelectedItem.text(0)
            Description = self.FactorDB.getTable(SelectedTableName).getMetaData(key="Description")
            return QMessageBox.information(self, "描述信息", str(Description))
        isFactorSelected, SelectedItem = self.isBottomItemSelected()
        if isFactorSelected:
            SelectedFactorName = SelectedItem.text(0)
            SelectedTableName = SelectedItem.parent().text(0)
            Description = self.FactorDB.getTable(SelectedTableName).getFactorMetaData([SelectedFactorName], key="Description").iloc[0]
            return QMessageBox.information(self, "描述信息", str(Description))
        return 0
    @pyqtSlot()
    def on_ViewButton_clicked(self):
        isFactorSelected, SelectedItem = self.isBottomItemSelected()
        if not isFactorSelected: return QMessageBox.critical(self, "错误", "请选择一个因子!")
        Factor = self.FactorDB.getTable(SelectedItem.parent().text(0)).getFactor(SelectedItem.text(0))
        Dlg = PreviewDlg(factor=Factor, parent=self)
        Dlg.exec_()
        return 0
    def renameTable(self, selected_item):
        OldTableName = selected_item.text(0)
        # 获取新表名
        NewTableName, isOk = QInputDialog.getText(self, "表名", "请输入表名", text=OldTableName)
        if (not isOk) or (OldTableName==NewTableName): return 0
        if NewTableName in self.FactorDB.TableNames:
            return QMessageBox.critical(self, "错误", "当前包含重名表!")
        # 调整其他关联区的数据
        try:
            self.FactorDB.renameTable(OldTableName, NewTableName)
            selected_item.setText(0, NewTableName)
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
        return 0
    def renameFactor(self, selected_item):
        OldFactorName = selected_item.text(0)
        TableName = selected_item.parent().text(0)
         # 获取新因子名
        NewFactorName, isOK = QInputDialog.getText(self, "因子名", "请输入因子名", text=OldFactorName)
        if (not isOK) or (OldFactorName==NewFactorName): return 0
        if NewFactorName in self.FactorDB.getTable(TableName).FactorNames:
            return QMessageBox.critical(self, "错误", "该表中包含重名因子!")
        try:
            self.FactorDB.renameFactor(TableName, OldFactorName, NewFactorName)
            selected_item.setText(0, NewFactorName)
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
        return 0
    @pyqtSlot()
    def on_RenameButton_clicked(self):
        # 检查用户是否选择了一张表
        isTableSelected, SelectedItem = self.isTopItemSelected()
        if isTableSelected: return self.renameTable(SelectedItem)
        # 检查用户是否选择了一个因子
        isFactorSelected, SelectedItem = self.isBottomItemSelected()
        if isFactorSelected: return self.renameFactor(SelectedItem)
        return 0
    @pyqtSlot()
    def on_DeleteButton_clicked(self):
        isOK = QMessageBox.question(self, "删除", "删除后将无法恢复, 你是否能对自己的行为负责?", QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel)
        if isOK!=QMessageBox.Ok: return 0
        TableFactor = self.genTableFactor()
        for iTable in TableFactor:
            try:
                if TableFactor[iTable] is None:
                    self.FactorDB.deleteTable(iTable)
                else:
                    self.FactorDB.deleteFactor(iTable, TableFactor[iTable])
            except Exception as e:
                QMessageBox.critical(self, "错误", str(e))
        return self.populateFactorDBTree()
    @pyqtSlot()
    def on_MoveButton_clicked(self):
        TableFactor = self.genTableFactor()
        if not TableFactor: return 0
        # 获取表名
        TableList = self.FactorDB.TableNames
        TargetTableName, isOk = QInputDialog.getItem(self, "目标表", "请选择目标表", ["新建..."]+TableList, editable=False)
        if not isOk: return 0
        if TargetTableName=="新建...":
            TargetTableName, isOk = QInputDialog.getText(self, "新建表", "请输入表名", text=genAvailableName("NewTable", TableList))
            if not isOk: return 0
            if TargetTableName in TableList:
                QMessageBox.critical(self, "错误", "当前包含重名表!")
                return 0
        self.setEnabled(False)
        for iTable, iFactorNames in TableFactor.items():
            iFT = self.FactorDB.getTable(iTable)
            if iFactorNames is None: iFactorNames = iFT.FactorNames
            iIDs, iDTs = iFT.getID(), iFT.getDataTime()
            iData = iFT.readData(factor_names=iFactorNames, ids=iIDs, dts=iDTs)
            self.FactorDB.writeData(iData, TargetTableName, if_exists="update")
        self.populateFactorDBTree()
        QMessageBox.information(self, "完成", "因子移动完成!")
        self.setEnabled(True)
        return 0
    @pyqtSlot()
    def on_CSVExportButton_clicked(self):
        DirPath = QFileDialog.getExistingDirectory(parent=self, caption="导出CSV", directory=os.getcwd())
        if not DirPath: return 0
        self.setEnabled(False)
        TableFactor = self.genTableFactor()
        for iTable, iFactorNames in TableFactor.items():
            iFT = self.FactorDB.getTable(iTable)
            if iFactorNames is None: iFactorNames = iFT.FactorNames
            iDTs, iIDs = iFT.getDateTime(), iFT.getID()
            iData = iFT.readData(factor_names=iFactorNames, ids=iIDs, dts=iDTs)
            for j, jFactorName in enumerate(iData.items):
                iData.iloc[j].to_csv(DirPath+os.sep+iTable+"-"+jFactorName+".csv", encoding="utf-8")
        QMessageBox.information(self, "完成", "导出数据完成!")
        self.setEnabled(True)
        return 0
    @pyqtSlot()
    def on_CSVImportButton_clicked(self):
        SelectedItems = self.FactorDBTree.selectedItems()
        nSelectedItems = len(SelectedItems)
        if (nSelectedItems>1):
            QMessageBox.critical(self, "错误", "请选择一张表或一个因子!")
            return 0
        elif nSelectedItems==0:
            # 获取新表名
            isOk, TableName = self.getNewTableName()
            if not isOk: return 0
            NewFactorName = "NewFactor"
        else:
            if SelectedItems[0].parent() is None:
                TableName = SelectedItems[0].text(0)
                isOk, NewFactorName = self.getNewFactorName(TableName)
                if not isOk: return 0
            else:
                TableName = SelectedItems[0].parent().text(0)
                NewFactorName = SelectedItems[0].text(0)
        FilePath = QFileDialog.getOpenFileName(parent=self, caption="导入CSV", directory=".", filter="csv (*.csv)")[0]
        if not FilePath: return 0
        if (TableName in self.FactorDB.TableNames) and (NewFactorName in self.FactorDB.getTable(TableName).FactorNames):
            if_exists, isOk = QInputDialog.getItem(self, "因子合并", "因子合并方式:", ["replace", "append", "update"], editable=False)
            if not isOk: return 0
        else:
            if_exists = "update"
        self.setEnabled(False)
        FactorData = loadCSVFactorData(FilePath)
        try:
            self.FactorDB.writeData(pd.Panel({NewFactorName:FactorData}), TableName, if_exists=if_exists)
            self.populateFactorDBTree()
            QMessageBox.information(self, '完成', '导入数据完成!')
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
        self.setEnabled(True)
        return 0

if __name__=='__main__':
    # 测试代码
    import sys
    from PyQt5.QtWidgets import QApplication
    import QuantStudio.api as QS
    HDB = QS.FactorDB.HDF5DB()
    HDB.connect()
    
    app = QApplication(sys.argv)
    TestWindow = FactorDBDlg(HDB)
    TestWindow.show()
    sys.exit(app.exec_())