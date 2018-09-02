# -*- coding: utf-8 -*-
import os
import shelve

from PyQt4.QtCore import pyqtSlot, SIGNAL, QDate
from PyQt4.QtGui import QDialog,QMessageBox,QListWidgetItem,QAction,QFileDialog
from .Ui_IDSetup import Ui_IDSetupDlg

from QuantStudio.FunLib.DateTimeFun import getCurrentDateStr,DateStr2Datetime,Datetime2DateStr
from QuantStudio.FunLib.FileFun import readCSV2Pandas
from QuantStudio.FunLib.IDFun import suffixAShareID

class IDSetupDlg(QDialog, Ui_IDSetupDlg):
    """ID设置对话框"""
    def __init__(self, parent=None, ids=[], qs_env=None):
        super().__init__(parent)
        self.setupUi(self)
        self.IDs = ids# 当前的ID序列
        self.ID1s = []
        self.ID2s = []
        self.CurInputID = ''
        self.QSEnv = qs_env
        self.isChanged = False
        self.MainDB = None
        self.IndexInfo = None
        with shelve.open(self.QSEnv.SysArgs['LibPath']+os.sep+"IndexInfo") as LibFile:
            for iDBName in self.QSEnv.SysArgs['DBList']:
                iDB = getattr(self.QSEnv,iDBName)
                if iDB.isAvailable():
                    self.MainDB = iDB
                    self.IndexInfo = LibFile.get(iDBName)
            if self.IndexInfo is None:
                self.IndexInfo = LibFile[self.QSEnv.SysArgs['DBList'][0]]
        self.AllIndexIDs = list(self.IndexInfo['成份股']['ID'])
        self.AllIndexNames = list(self.IndexInfo['成份股']['指数名称'])
        if self.MainDB is None:
            self.HistoryRadioButton.setEnabled(False)
            self.HistoryRadioButton.setChecked(False)
            self.CurrentRadioButton.setChecked(True)
            self.SelectID1Button.setEnabled(False)
            self.SelectID2Button.setEnabled(False)
            self.IndexNameEdit.setEnabled(False)
            self.IndexIDEdit.setEnabled(False)
            self.DateCalendar.setEnabled(False)
            self.DateEdit.setEnabled(False)
            self.CurrentRadioButton.setEnabled(False)
        self.DateEdit.setText(getCurrentDateStr())
        self.IndexNameEdit.setText('全体A股')
        # 设置IDListWidget的弹出菜单
        NewAction = QAction('删除选中项', self.IDListWidget)
        self.connect(NewAction,SIGNAL("triggered()"), self.deleteIDListItems)
        self.IDListWidget.addAction(NewAction)
        NewAction = QAction('清空', self.IDListWidget)
        self.connect(NewAction, SIGNAL("triggered()"), self.clearIDList)
        self.IDListWidget.addAction(NewAction)
        NewAction = QAction('复制 ID', self.IDListWidget)
        self.connect(NewAction, SIGNAL("triggered()"), self.copyID)
        self.IDListWidget.addAction(NewAction)
        NewAction = QAction('粘贴 ID', self.IDListWidget)
        self.connect(NewAction, SIGNAL("triggered()"), self.pasteID)
        self.IDListWidget.addAction(NewAction)
        NewAction = QAction('导入 ID',self.IDListWidget)
        self.connect(NewAction,SIGNAL("triggered()"),self.importData)
        self.IDListWidget.addAction(NewAction)
        NewAction = QAction('导出 ID',self.IDListWidget)
        self.connect(NewAction,SIGNAL("triggered()"),self.exportData)
        self.IDListWidget.addAction(NewAction)
        self.populateIDListWidget(self.IDs)
        # 设置指数信息更新菜单
        NewAction = QAction('更新指数信息',self.IndexIDEdit)
        self.connect(NewAction,SIGNAL("triggered()"),self.updateIndexInfo)
        self.IndexIDEdit.addAction(NewAction)
        NewAction = QAction('更新指数信息',self.IndexNameEdit)
        self.connect(NewAction,SIGNAL("triggered()"),self.updateIndexInfo)
        self.IndexNameEdit.addAction(NewAction)
        # 设置本地因子数据库信息
        self.FDBComboBox.blockSignals(True)
        self.FTComboBox.blockSignals(True)
        self.FDBComboBox.addItems(['FactorDB']+self.QSEnv.SysArgs['FactorDBList'])
        FactorDB = getattr(self.QSEnv,"FactorDB")
        TableList = FactorDB.TableNames
        TableList.sort()
        self.FTComboBox.addItems(TableList)
        self.FDBComboBox.blockSignals(False)
        self.FTComboBox.blockSignals(False)
        FactorList = FactorDB.getFactorName(TableList[0])
        FactorList.sort()
        self.FComboBox.addItems(FactorList)
        # 内部版设置
        #if not self.QSEnv.SysArgs['isInsider']:
            #self.label.setVisible(False)
            #self.DateEdit.setVisible(False)
            #self.DateCalendar.setVisible(False)
            #self.CurrentRadioButton.setVisible(False)
            #self.HistoryRadioButton.setVisible(False)
            #self.label_2.setVisible(False)
            #self.label_3.setVisible(False)
            #self.IndexIDEdit.setVisible(False)
            #self.IndexNameEdit.setVisible(False)
            #self.SelectID1Button.setVisible(False)
            #self.SelectID2Button.setVisible(False)
        return
    def updateIndexInfo(self):
        self.setEnabled(False)
        with shelve.open(self.QSEnv.SysArgs['LibPath']+os.sep+"IndexInfo") as LibFile:
            if self.QSEnv.WindDB.isAvailable():
                self.IndexInfo = self.QSEnv.WindDB.getIndexInfo()
                LibFile['WindDB'] = self.IndexInfo
            elif self.QSEnv.WindDB2.isAvailable():
                self.IndexInfo = self.QSEnv.WindDB2.getIndexInfo()
                LibFile['WindDB2'] = self.IndexInfo
            else:
                QMessageBox.information(None,'失败','没有连接数据库,更新失败!')
                self.setEnabled(True)
                return 0
        self.populateIndexTree()
        QMessageBox.information(None,'完成','指数信息更新完成!')
        self.setEnabled(True)
        return 0
    def populateIDListWidget(self, ids):
        self.IDListWidget.clear()
        self.IDListWidget.addItems(ids)
        self.IDNumEdit.setText(str(len(ids)))
        return 0
    def populateID1ListWidget(self, ids):
        self.ID1ListWidget.clear()
        self.ID1ListWidget.addItems(ids)
        return 0
    def populateID2ListWidget(self, ids):
        self.ID2ListWidget.clear()
        self.ID2ListWidget.addItems(ids)
        return 0
    def deleteIDListItems(self):
        for iItem in self.IDListWidget.selectedItems():
            self.IDs.remove(iItem.text())
        self.populateIDListWidget(self.IDs)
        return 0
    def clearIDList(self):
        self.IDs = []
        self.IDListWidget.clear()
        self.IDNumEdit.setText("0")
        return 0
    def copyID(self):
        self.QSEnv.Clipboard['IDs'] = self.IDs
        return 0
    def pasteID(self):
        IDs = self.QSEnv.Clipboard.get("IDs")
        if IDs is None:
            QMessageBox.critical(None,"错误","剪贴板中没有复制过的ID序列!")
            return 0
        self.IDs = IDs
        self.populateIDListWidget(self.IDs)
        return 0
    @pyqtSlot()
    def on_DateEdit_editingFinished(self):
        Date = DateStr2Datetime(self.DateEdit.text())
        if Date is None:
            QMessageBox.critical(None,'错误','日期格式不正确!')
            self.DateEdit.setText(Datetime2DateStr(self.DateCalendar.selectedDate().toPyDate()))
            return 0
        self.DateCalendar.setSelectedDate(QDate(Date.year, Date.month, Date.day))
        return 0
    @pyqtSlot()
    def on_IndexNameEdit_editingFinished(self):
        IndexName = self.IndexNameEdit.text()
        try:
            tempInd = self.AllIndexNames.index(IndexName)
            IndexID = self.AllIndexIDs[tempInd]
            self.IndexIDEdit.blockSignals(True)
            self.IndexIDEdit.setText(IndexID)
            self.IndexIDEdit.blockSignals(False)
        except:
            pass
        return 0
    @pyqtSlot()
    def on_IndexIDEdit_editingFinished(self):
        IndexID = self.IndexIDEdit.text()
        self.IndexNameEdit.blockSignals(True)
        if IndexID == '':
            self.IndexNameEdit.setText('全体A股')
            self.IndexNameEdit.blockSignals(False)
            return 0
        try:
            tempInd = self.AllIndexIDs.index(IndexID)
            IndexName = self.AllIndexNames[tempInd]
            self.IndexNameEdit.setText(IndexName)
            self.IndexNameEdit.blockSignals(False)
            return 0
        except:
            if self.QSEnv.WindAddin.isAvailable():
                IndexName = self.QSEnv.WindAddin.getSecName(IndexID)
                if IndexName is not None:
                    self.IndexNameEdit.setText(IndexName)
                else:
                    self.IndexNameEdit.setText('')
        self.IndexNameEdit.blockSignals(False)
        return 0
    @pyqtSlot()
    def on_AcceptButton_clicked(self):
        self.close()
        self.isChanged = True
        self.IDInputEdit.releaseKeyboard()
        return True
    @pyqtSlot()
    def on_RejectButton_clicked(self):
        self.close()
        self.IDInputEdit.releaseKeyboard()
        return False
    @pyqtSlot()
    def on_DateCalendar_selectionChanged(self):
        Date = Datetime2DateStr(self.DateCalendar.selectedDate().toPyDate())
        self.DateEdit.setText(Date)
        return 0
    @pyqtSlot(bool)
    def on_HistoryRadioButton_toggled(self, checked):
        if checked:
            self.CurrentRadioButton.setChecked(False)
        return 0
    @pyqtSlot(bool)
    def on_CurrentRadioButton_toggled(self, checked):
        if checked:
            self.HistoryRadioButton.setChecked(False)
        return 0
    def exportData(self):
        if len(self.IDs)==0:
            return 0
        FilePath = QFileDialog.getSaveFileName(None,'导出数据','..\\ID.csv',"Excel (*.csv)")
        if len(FilePath)==0:
            return 0
        import pandas as pd
        temp = pd.DataFrame(self.IDs,columns=['ID'])
        temp.to_csv(FilePath,header=False,index=False)
        QMessageBox.information(None,'完成','导出数据完成!')
        return 0
    def importData(self):
        FilePath = QFileDialog.getOpenFileName(None,'导入数据','..',"Excel (*.csv)")
        if len(FilePath)==0:
            return 0
        IDs = readCSV2Pandas(FilePath,detect_file_encoding=True,index_col=None,header=None)
        self.IDs = list(IDs.values[:,0])
        self.populateIDListWidget(self.IDs)
        return 0
    @pyqtSlot()
    def on_IDInputEdit_returnPressed(self):
        ID = self.IDInputEdit.text()
        if ID=='':
            return 0
        self.IDs = set(self.IDs)
        self.IDs.add(ID)
        self.IDs = list(self.IDs)
        self.IDs.sort()
        self.populateIDListWidget(self.IDs)
        self.IDInputEdit.setText('')
        self.CurInputID = ''
        return 0
    @pyqtSlot(str)
    def on_IDInputEdit_textEdited(self, p0):
        if p0.strip(self.CurInputID)=='.':
            self.IDInputEdit.setText(suffixAShareID(p0[:-1]))
        self.CurInputID = p0
        return 0
    # 提取成分股
    def selectID(self):
        Date = self.DateEdit.text()
        IndexName = self.IndexNameEdit.text()
        IndexID = self.IndexIDEdit.text()
        isCurrent = self.CurrentRadioButton.isChecked()
        if IndexName=='全体A股':
            if self.MainDB is not None:
                IDs = self.MainDB.getID(IndexID="全体A股", date=Date, is_current=isCurrent)
            else:
                return (-1,[])
        else:
            try:
                IDs = self.MainDB.getID(IndexID, date=Date, is_current=isCurrent)
            except:
                IDs = []
            if IDs==[]:
                QMessageBox.critical(None,'错误','提取成分股失败!\n可能的原因:\n(1)指数名称或者代码不正确;\n(2)该日期处指数不存在;\n(3)该指数系统不支持.')
                return (-1,[])
        IDs.sort()
        return (1,IDs)
    @pyqtSlot()
    def on_SelectID1Button_clicked(self):
        Res = self.selectID()
        if Res[0]==-1:
            return 0
        else:
            self.ID1s = Res[1]
            self.populateID1ListWidget(self.ID1s)
            return 0
    @pyqtSlot()
    def on_IDToID1Button_clicked(self):
        self.ID1s = self.IDs
        self.populateID1ListWidget(self.ID1s)
        return 0
    @pyqtSlot()
    def on_SelectIDButton_clicked(self):
        CombineType = self.CombineComboBox.currentText()
        if CombineType=='并':
            self.IDs = list(set(self.ID1s).union(self.ID2s))
        elif CombineType=='交':
            self.IDs = list(set(self.ID1s).intersection(self.ID2s))
        elif CombineType=='上差下':
            self.IDs = list(set(self.ID1s).difference(self.ID2s))
        elif CombineType=='下差上':
            self.IDs = list(set(self.ID2s).difference(self.ID1s))
        elif CombineType=='对称差':
            self.IDs = list(set(self.ID1s).symmetric_difference(self.ID2s))
        elif CombineType=='只取上':
            self.IDs = self.ID1s
        elif CombineType=='只取下':
            self.IDs = self.ID2s
        self.IDs.sort()
        self.populateIDListWidget(self.IDs)
        return 0
    @pyqtSlot()
    def on_SelectID2Button_clicked(self):
        Res = self.selectID()
        if Res[0]==-1:
            return 0
        else:
            self.ID2s = Res[1]
            self.populateID2ListWidget(self.ID2s)
            return 0
    @pyqtSlot()
    def on_IDToID2Button_clicked(self):
        self.ID2s = self.IDs
        self.populateID2ListWidget(self.ID2s)
        return 0
    @pyqtSlot()
    def on_LocalSelectID2Button_clicked(self):
        FactorDB = getattr(self.QSEnv,self.FDBComboBox.currentText())
        TableName = self.FTComboBox.currentText()
        FactorName = self.FComboBox.currentText()
        self.ID2s = FactorDB.getID(TableName,FactorName)
        self.populateID2ListWidget(self.ID2s)
        return 0
    @pyqtSlot(str)
    def on_FDBComboBox_currentIndexChanged(self, p0):
        FactorDB = getattr(self.QSEnv,p0)
        TableList = FactorDB.TableNames
        TableList.sort()
        self.FTComboBox.blockSignals(True)
        self.FTComboBox.clear()
        self.FTComboBox.addItems(TableList)
        self.FTComboBox.blockSignals(False)
        self.FComboBox.clear()
        FactorList = FactorDB.getFactorName(TableList[0])
        FactorList.sort()
        self.FComboBox.addItems(FactorList)
        return 0
    @pyqtSlot(str)
    def on_FTComboBox_currentIndexChanged(self, p0):
        FactorDB = getattr(self.QSEnv,self.FDBComboBox.currentText())
        self.FComboBox.clear()
        FactorList = FactorDB.getFactorName(p0)
        FactorList.sort()
        self.FComboBox.addItems(FactorList)
        return 0