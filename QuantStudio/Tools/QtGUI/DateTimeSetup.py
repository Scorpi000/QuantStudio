# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from QuantStudio.GUI.Ui_DateTimeSetup import Ui_Dialog

from QuantStudio.FunLib.DateTimeFun import getCurrentDateStr,DateStr2Datetime,Datetime2DateStr,getNaturalDay,getMonthFirstDay,getMonthMiddleDay,getMonthLastDay,getWeekFirstDay,getWeekLastDay,getQuarterFirstDay,getQuarterLastDay,getFinancialQuarterFirstDay,getFinancialQuarterLastDay,getYearFirstDay,getYearLastDay
from QuantStudio.FunLib.FileFun import readCSV2Pandas


class DateTimeSetupDlg(QDialog, Ui_Dialog):
    """日期时间设置对话框"""
    def __init__(self, parent=None, dates=[], qs_env=None):
        super(DateTimeSetupDlg, self).__init__(parent)
        self.setupUi(self)
        
        self.Dates = dates.copy()
        self.Date1s = []
        self.Date2s = []
        self.QSEnv = qs_env
        self.isChanged = False
        NowDate = getCurrentDateStr()
        self.StartDateEdit.setText(NowDate)
        self.EndDateEdit.setText(NowDate)
        self.DateTypeComboBox.addItems(['交易日','自然日'])
        self.DateTransformComboBox.addItems(['连续日期', '月末日', '周末日', '年末日','季末日','财报季末日','月初日','周初日','年初日','季初日','财报季初日','月中日'])# 初始化日期变换ComboBox
        # 初始化Calendar
        NowDate = DateStr2Datetime(NowDate)
        self.EndDateCalendar.setSelectedDate(QDate(NowDate.year, NowDate.month, NowDate.day))
        self.StartDateCalendar.setSelectedDate(QDate(NowDate.year, NowDate.month, NowDate.day))
        # 设置DateListWidget的弹出菜单
        self.DateListWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.DateListWidget.customContextMenuRequested.connect(self.showDateListWidgetContextMenu)
        self.DateListWidget.ContextMenu = {'主菜单':QMenu()}
        NewAction = self.DateListWidget.ContextMenu['主菜单'].addAction('删除选中项')
        NewAction.triggered.connect(self.deleteDateListItems)
        NewAction = self.DateListWidget.ContextMenu['主菜单'].addAction('清空')
        NewAction.triggered.connect(self.clearDateList)
        NewAction = self.DateListWidget.ContextMenu['主菜单'].addAction('复制日期')
        NewAction.triggered.connect(self.copyDate)
        NewAction = self.DateListWidget.ContextMenu['主菜单'].addAction('粘贴日期')
        NewAction.triggered.connect(self.pasteDate)
        self.DateListWidget.ContextMenu['导入导出'] = {"主菜单":QMenu("导入导出")}
        NewAction = self.DateListWidget.ContextMenu['导入导出']['主菜单'].addAction('导入日期')
        NewAction.triggered.connect(self.importData)
        NewAction = self.DateListWidget.ContextMenu['导入导出']['主菜单'].addAction('导出日期')
        NewAction.triggered.connect(self.exportData)
        self.DateListWidget.ContextMenu['主菜单'].addMenu(self.DateListWidget.ContextMenu['导入导出']['主菜单'])
        self.DateListWidget.ContextMenu['日期变换'] = {"主菜单":QMenu("日期变换")}
        NewAction = self.DateListWidget.ContextMenu['日期变换']['主菜单'].addAction('转月底')
        NewAction.triggered.connect(self.toMonthLastDay)
        NewAction = self.DateListWidget.ContextMenu['日期变换']['主菜单'].addAction('转月中')
        NewAction.triggered.connect(self.toMonthMiddleDay)
        NewAction = self.DateListWidget.ContextMenu['日期变换']['主菜单'].addAction('转月初')
        NewAction.triggered.connect(self.toMonthFirstDay)
        NewAction = self.DateListWidget.ContextMenu['日期变换']['主菜单'].addAction('转周末')
        NewAction.triggered.connect(self.toWeekLastDay)
        NewAction = self.DateListWidget.ContextMenu['日期变换']['主菜单'].addAction('转周初')
        NewAction.triggered.connect(self.toWeekFirstDay)
        NewAction = self.DateListWidget.ContextMenu['日期变换']['主菜单'].addAction('转年末')
        NewAction.triggered.connect(self.toYearLastDay)
        NewAction = self.DateListWidget.ContextMenu['日期变换']['主菜单'].addAction('转年初')
        NewAction.triggered.connect(self.toYearFirstDay)
        NewAction = self.DateListWidget.ContextMenu['日期变换']['主菜单'].addAction('转月初')
        NewAction.triggered.connect(self.toMonthFirstDay)
        NewAction = self.DateListWidget.ContextMenu['日期变换']['主菜单'].addAction('转季末')
        NewAction.triggered.connect(self.toQuarterLastDay)
        NewAction = self.DateListWidget.ContextMenu['日期变换']['主菜单'].addAction('转季初')
        NewAction.triggered.connect(self.toQuarterFirstDay)
        NewAction = self.DateListWidget.ContextMenu['日期变换']['主菜单'].addAction('转财报季末')
        NewAction.triggered.connect(self.toFinancialQuarterLastDay)
        NewAction = self.DateListWidget.ContextMenu['日期变换']['主菜单'].addAction('转财报季初')
        NewAction.triggered.connect(self.toFinancialQuarterFirstDay)
        self.DateListWidget.ContextMenu['主菜单'].addMenu(self.DateListWidget.ContextMenu['日期变换']['主菜单'])
        NewAction = self.DateListWidget.ContextMenu['主菜单'].addAction('等间隔抽样')
        NewAction.triggered.connect(self.sampleDate)
        self.DateListWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.populateDateListWidget(self.Dates)
        #self.DateInputEdit.grabKeyboard()
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
            #self.StartDateCalendar.setVisible(False)
            #self.StartDateEdit.setVisible(False)
            #self.EndDateCalendar.setVisible(False)
            #self.EndDateEdit.setVisible(False)
            #self.label_2.setVisible(False)
            #self.label_3.setVisible(False)
            #self.label_4.setVisible(False)
            #self.DateTypeComboBox.setVisible(False)
            #self.DateTransformComboBox.setVisible(False)
            #self.SelectDate1Button.setVisible(False)
            #self.SelectDate2Button.setVisible(False)        
    
    @pyqtSlot()
    def on_AcceptButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError
    
    @pyqtSlot(str)
    def on_FDBComboBox_currentTextChanged(self, p0):
        """
        Slot documentation goes here.
        
        @param p0 DESCRIPTION
        @type str
        """
        # TODO: not implemented yet
        raise NotImplementedError
    
    @pyqtSlot(str)
    def on_FTComboBox_currentTextChanged(self, p0):
        """
        Slot documentation goes here.
        
        @param p0 DESCRIPTION
        @type str
        """
        # TODO: not implemented yet
        raise NotImplementedError
    
    @pyqtSlot()
    def on_SelectFDateTimeButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError
    
    @pyqtSlot()
    def on_RejectButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError
    
    @pyqtSlot(str)
    def on_DateSouceComboBox_currentTextChanged(self, p0):
        """
        Slot documentation goes here.
        
        @param p0 DESCRIPTION
        @type str
        """
        # TODO: not implemented yet
        raise NotImplementedError
    
    @pyqtSlot()
    def on_SelectDateButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError
    
    @pyqtSlot(str)
    def on_TimeSourceComboBox_currentTextChanged(self, p0):
        """
        Slot documentation goes here.
        
        @param p0 DESCRIPTION
        @type str
        """
        # TODO: not implemented yet
        raise NotImplementedError
    
    @pyqtSlot()
    def on_SelectTimeButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError
    
    @pyqtSlot()
    def on_CombineButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError
