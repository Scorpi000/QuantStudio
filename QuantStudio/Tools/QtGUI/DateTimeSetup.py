# -*- coding: utf-8 -*-
import os
import datetime as dt

import numpy as np
import pandas as pd
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QDialog, QMessageBox, QInputDialog, QFileDialog, QAction, QMenu
from QuantStudio.Tools.QtGUI.Ui_DateTimeSetup import Ui_Dialog

from QuantStudio.FactorDataBase.FactorDB import FactorTable
from QuantStudio.Tools import DateTimeFun
from QuantStudio.Tools.FileFun import readCSV2Pandas

def mergeSet(set1, set2, merge_type):
    if callable(merge_type): return merge_type(set1, set2)
    elif merge_type=="并集": return set1.union(set2)
    elif merge_type=="交集": return set1.intersection(set2)
    elif merge_type=="左差右": return set1.difference(set2)
    elif merge_type=="右差左": return set2.difference(set1)
    elif merge_type=="对称差": return set1.symmetric_difference(set2)
    return set1

class DateTimeSetupDlg(QDialog, Ui_Dialog):
    """日期时间设置对话框"""
    def __init__(self, parent=None, dts=[], dates=[], times=[], ft=None):
        super(DateTimeSetupDlg, self).__init__(parent)
        self.setupUi(self)
        self.DateTimes = dts.copy()
        self.Dates = dates.copy()
        self.Times = times.copy()
        self.isChanged = False
        # 设置因子库信息
        self.StartDTEdit.setDateTime(dt.datetime.combine(dt.date.today()-dt.timedelta(365), dt.time(0)))
        self.EndDTEdit.setDateTime(dt.datetime.today())
        if isinstance(ft, FactorTable):
            self.FT = ft
            self.FDB = self.FT.FactorDB
            TableNames = (self.FDB.TableNames if self.FDB is not None else [self.FT.Name])
            self.FDBGroupBox.setTitle("来自: "+self.FDB.Name)
            self.FTComboBox.blockSignals(True)
            self.FTComboBox.addItems(TableNames)
            self.FTComboBox.setCurrentText(self.FT.Name)
            self.FTComboBox.blockSignals(False)
            self.FactorComboBox.addItems(self.FT.FactorNames)
        else:
            self.FDB = self.FT = None
            self.FDBGroupBox.setEnabled(False)
        # 初始化日期设置
        self.StartDateEdit.setDate(dt.date.today()-dt.timedelta(365))
        self.EndDateEdit.setDate(dt.date.today())
        if hasattr(self.FDB, "getTradeDay"): self.DateTypeComboBox.addItems(["交易日", "自然日"])
        else: self.DateTypeComboBox.addItems(["自然日"])
        self.DatePeriodComboBox.addItems(["连续日期", "周末日", "月末日", "季末日", "财报季末日", "年末日", "周初日", "月初日", "季初日", "财报季初日", "年初日", "月中日"])
        self.populateDateListWidget(self.Dates)
        self.setDateListWidgetMenu()
        # 初始化时间设置
        self.populateTimeListWidget(self.Times)
        self.setTimeListWidgetMenu()
        # 初始化时点设置
        self.populateDateTimeListWidget(self.DateTimes)
        self.setDateTimeListWidgetMenu()
    @pyqtSlot()
    def on_AcceptButton_clicked(self):
        self.close()
        self.isChanged = True
        return True
    @pyqtSlot()
    def on_RejectButton_clicked(self):
        self.close()
        return False
    def deleteItems(self):
        ListWidget = self.sender().parent().parent()
        PopIndexes = [iIndex.row() for iIndex in ListWidget.selectedIndexes()]
        if ListWidget is self.DateListWidget:
            self.Dates = np.array(self.Dates, dtype="O")[sorted(set(np.arange(len(self.Dates))).difference(PopIndexes))].tolist()
            self.populateDateListWidget(self.Dates)
        elif ListWidget is self.TimeListWidget:
            self.Times = np.array(self.Times, dtype="O")[sorted(set(np.arange(len(self.Times))).difference(PopIndexes))].tolist()
            self.populateTimeListWidget(self.Times)
        elif ListWidget is self.DateTimeListWidget:
            self.DateTimes = np.array(self.DateTimes, dtype="O")[sorted(set(np.arange(len(self.DateTimes))).difference(PopIndexes))].tolist()
            self.populateDateTimeListWidget(self.DateTimes)
        return 0
    def clearList(self):
        ListWidget = self.sender().parent().parent()
        if ListWidget is self.DateListWidget: self.Dates = []
        elif ListWidget is self.TimeListWidget: self.Times = []
        elif ListWidget is self.DateTimeListWidget:
            self.DateTimes = []
            self.DateTimeNumEdit.setText("0")
        ListWidget.clear()
        return 0
    def _changeDT(self, dt_period):
        ListWidget = self.sender().parent().parent().parent()
        if ListWidget is self.DateListWidget:
            self.Dates = self.changeDateTime(self.Dates, dt_period=dt_period)
            self.populateDateListWidget(self.Dates)
        elif ListWidget is self.DateTimeListWidget:
            self.DateTimes = self.changeDateTime(self.DateTimes, dt_period=dt_period)
            self.populateDateTimeListWidget(self.DateTimes)
        return 0
    def toMonthFirstDay(self):
        return self._changeDT(dt_period="月初日")
    def toMonthMiddleDay(self):
        return self._changeDT(dt_period="月中日")
    def toMonthLastDay(self):
        return self._changeDT(dt_period="月末日")
    def toWeekFirstDay(self):
        return self._changeDT(dt_period="周初日")
    def toWeekLastDay(self):
        return self._changeDT(dt_period="周末日")
    def toYearFirstDay(self):
        return self._changeDT(dt_period="年初日")
    def toYearLastDay(self):
        return self._changeDT(dt_period="年末日")
    def toQuarterFirstDay(self):
        return self._changeDT(dt_period="季初日")
    def toQuarterLastDay(self):
        return self._changeDT(dt_period="季末日")
    def toFinancialQuarterFirstDay(self):
        return self._changeDT(dt_period="财报季初日")
    def toFinancialQuarterLastDay(self):
        return self._changeDT(dt_period="财报季末日")
    def sampleData(self):
        SamplePeriod, isOk = QInputDialog.getInt(self, "间隔", "采样间隔", 10, 0, 2147483647, 1)
        if not isOk: return 0
        ListWidget = self.sender().parent().parent().parent()
        if ListWidget is self.DateListWidget:
            self.Dates = [self.Dates[i] for i in range(0, len(self.Dates), SamplePeriod)]
            self.populateDateListWidget(self.Dates)
        elif ListWidget is self.TimeListWidget:
            self.Times = [self.Times[i] for i in range(0, len(self.Times), SamplePeriod)]
            self.populateTimeListWidget(self.Times)
        elif ListWidget is self.DateTimeListWidget:
            self.DateTimes = [self.DateTimes[i] for i in range(0, len(self.DateTimes), SamplePeriod)]
            self.populateDateTimeListWidget(self.DateTimes)
        return 0
    def exportData(self):
        ListWidget = self.sender().parent().parent().parent()
        if ListWidget is self.DateListWidget: Data = self.Dates
        elif ListWidget is self.TimeListWidget: Data = self.Times
        elif ListWidget is self.DateTimeListWidget: Data = self.DateTimes
        if not Data: return 0
        FilePath, _ = QFileDialog.getSaveFileName(self, "导出数据", os.getcwd()+os.sep+"untitled.csv", "csv (*.csv)")
        if not FilePath: return 0
        Data = pd.DataFrame(Data, columns=["Date"])
        Data.to_csv(FilePath, header=False, index=False)
        return QMessageBox.information(self, "完成", "导出完成!")
    def importData(self):
        FilePath, _ = QFileDialog.getOpenFileName(self, "导入数据", os.getcwd(), "csv (*.csv)")
        if not FilePath: return 0
        try:
            Data = readCSV2Pandas(FilePath, detect_file_encoding=True, index_col=None, header=None, parse_dates=True, infer_datetime_format=True)
        except Exception as e:
            return QMessageBox.critical(self, "错误", "数据读取失败: "+str(e))
        ListWidget = self.sender().parent().parent().parent()
        if ListWidget is self.DateListWidget: FormatStr = "%Y-%m-%d"
        elif ListWidget is self.TimeListWidget: FormatStr = "%H:%M:%S"
        elif ListWidget is self.DateTimeListWidget: FormatStr = "%Y-%m-%d %H:%M:%S.%f"
        FormatStr, isOk = QInputDialog.getText(self, "时间格式", "请输入文件的时间格式: ", text=FormatStr)
        if not isOk: return 0
        try:
            if ListWidget is self.DateListWidget:
                self.Dates = [dt.datetime.strptime(Data.iloc[i, 0], FormatStr).date() for i in range(Data.shape[0])]
                return self.populateDateListWidget(self.Dates)
            elif ListWidget is self.TimeListWidget:
                self.Times = [dt.datetime.strptime(Data.iloc[i, 0], FormatStr).time() for i in range(Data.shape[0])]
                return self.populateTimeListWidget(self.Times)
            elif ListWidget is self.DateTimeListWidget:
                self.DateTimes = [dt.datetime.strptime(Data.iloc[i, 0], FormatStr) for i in range(Data.shape[0])]
                return self.populateDateTimeListWidget(self.DateTimes)
        except Exception as e:
            return QMessageBox.critical(self, "错误", "数据解析失败: "+str(e))
    def showDateListWidgetMenu(self, pos):
        self.DateListWidget.ContextMenu["主菜单"].move(QCursor.pos())
        self.DateListWidget.ContextMenu["主菜单"].show()
    def setDateListWidgetMenu(self):
        # 设置 DateListWidget 的弹出菜单
        self.DateListWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.DateListWidget.customContextMenuRequested.connect(self.showDateListWidgetMenu)
        self.DateListWidget.ContextMenu = {"主菜单": QMenu(parent=self.DateListWidget)}
        self.DateListWidget.ContextMenu["主菜单"].addAction("删除选中项").triggered.connect(self.deleteItems)
        self.DateListWidget.ContextMenu["主菜单"].addAction("清空").triggered.connect(self.clearList)
        self.DateListWidget.ContextMenu["日期变换"] = {"主菜单": QMenu("日期变换", parent=self.DateListWidget.ContextMenu["主菜单"])}
        self.DateListWidget.ContextMenu["日期变换"]["主菜单"].addAction("转月底").triggered.connect(self.toMonthLastDay)
        self.DateListWidget.ContextMenu["日期变换"]["主菜单"].addAction("转月中").triggered.connect(self.toMonthMiddleDay)
        self.DateListWidget.ContextMenu["日期变换"]["主菜单"].addAction("转月初").triggered.connect(self.toMonthFirstDay)
        self.DateListWidget.ContextMenu["日期变换"]["主菜单"].addAction("转周末").triggered.connect(self.toWeekLastDay)
        self.DateListWidget.ContextMenu["日期变换"]["主菜单"].addAction("转周初").triggered.connect(self.toWeekFirstDay)
        self.DateListWidget.ContextMenu["日期变换"]["主菜单"].addAction("转年末").triggered.connect(self.toYearLastDay)
        self.DateListWidget.ContextMenu["日期变换"]["主菜单"].addAction("转年初").triggered.connect(self.toYearFirstDay)
        self.DateListWidget.ContextMenu["日期变换"]["主菜单"].addAction("转季末").triggered.connect(self.toQuarterLastDay)
        self.DateListWidget.ContextMenu["日期变换"]["主菜单"].addAction("转季初").triggered.connect(self.toQuarterFirstDay)
        self.DateListWidget.ContextMenu["日期变换"]["主菜单"].addAction("转财报季末").triggered.connect(self.toFinancialQuarterLastDay)
        self.DateListWidget.ContextMenu["日期变换"]["主菜单"].addAction("转财报季初").triggered.connect(self.toFinancialQuarterFirstDay)
        self.DateListWidget.ContextMenu["日期变换"]["主菜单"].addAction("等间隔抽样").triggered.connect(self.sampleData)
        self.DateListWidget.ContextMenu["主菜单"].addMenu(self.DateListWidget.ContextMenu["日期变换"]["主菜单"])
        self.DateListWidget.ContextMenu["导入导出"] = {"主菜单": QMenu("导入导出", parent=self.DateListWidget.ContextMenu["主菜单"])}
        self.DateListWidget.ContextMenu["导入导出"]["主菜单"].addAction("导入日期").triggered.connect(self.importData)
        self.DateListWidget.ContextMenu["导入导出"]["主菜单"].addAction("导出日期").triggered.connect(self.exportData)
        self.DateListWidget.ContextMenu["主菜单"].addMenu(self.DateListWidget.ContextMenu["导入导出"]["主菜单"])
        return 0
    def populateDateListWidget(self, dates):
        self.DateListWidget.clear()
        self.DateListWidget.addItems([iDate.strftime("%Y-%m-%d") for iDate in dates])
        return 0
    def showTimeListWidgetMenu(self, pos):
        self.TimeListWidget.ContextMenu["主菜单"].move(QCursor.pos())
        self.TimeListWidget.ContextMenu["主菜单"].show()
    def setTimeListWidgetMenu(self):
        # 设置 TimeListWidget 的弹出菜单
        self.TimeListWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.TimeListWidget.customContextMenuRequested.connect(self.showTimeListWidgetMenu)
        self.TimeListWidget.ContextMenu = {"主菜单": QMenu(parent=self.TimeListWidget)}
        self.TimeListWidget.ContextMenu["主菜单"].addAction("删除选中项").triggered.connect(self.deleteItems)
        self.TimeListWidget.ContextMenu["主菜单"].addAction("清空").triggered.connect(self.clearList)
        self.TimeListWidget.ContextMenu["时间变换"] = {"主菜单": QMenu("时间变换", parent=self.TimeListWidget.ContextMenu["主菜单"])}
        self.TimeListWidget.ContextMenu["时间变换"]["主菜单"].addAction("等间隔抽样").triggered.connect(self.sampleData)
        self.TimeListWidget.ContextMenu["主菜单"].addMenu(self.TimeListWidget.ContextMenu["时间变换"]["主菜单"])
        self.TimeListWidget.ContextMenu["导入导出"] = {"主菜单": QMenu("导入导出", parent=self.TimeListWidget.ContextMenu["主菜单"])}
        self.TimeListWidget.ContextMenu["导入导出"]["主菜单"].addAction("导入时间").triggered.connect(self.importData)
        self.TimeListWidget.ContextMenu["导入导出"]["主菜单"].addAction("导出时间").triggered.connect(self.exportData)
        self.TimeListWidget.ContextMenu["主菜单"].addMenu(self.TimeListWidget.ContextMenu["导入导出"]["主菜单"])
        return 0
    def populateTimeListWidget(self, times):
        self.TimeListWidget.clear()
        self.TimeListWidget.addItems([iTime.strftime("%H:%M:%S") for iTime in times])
        return 0
    def showDateTimeListWidgetMenu(self, pos):
        self.DateTimeListWidget.ContextMenu["主菜单"].move(QCursor.pos())
        self.DateTimeListWidget.ContextMenu["主菜单"].show()
    def setDateTimeListWidgetMenu(self):
        # 设置 DateTimeListWidget 的弹出菜单
        self.DateTimeListWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.DateTimeListWidget.customContextMenuRequested.connect(self.showDateTimeListWidgetMenu)
        self.DateTimeListWidget.ContextMenu = {"主菜单": QMenu(parent=self.DateTimeListWidget)}
        self.DateTimeListWidget.ContextMenu["主菜单"].addAction("删除选中项").triggered.connect(self.deleteItems)
        self.DateTimeListWidget.ContextMenu["主菜单"].addAction("清空").triggered.connect(self.clearList)
        self.DateTimeListWidget.ContextMenu["时点变换"] = {"主菜单": QMenu("时点变换", parent=self.DateTimeListWidget.ContextMenu["主菜单"])}
        self.DateTimeListWidget.ContextMenu["时点变换"]["主菜单"].addAction("转月底").triggered.connect(self.toMonthLastDay)
        self.DateTimeListWidget.ContextMenu["时点变换"]["主菜单"].addAction("转月中").triggered.connect(self.toMonthMiddleDay)
        self.DateTimeListWidget.ContextMenu["时点变换"]["主菜单"].addAction("转月初").triggered.connect(self.toMonthFirstDay)
        self.DateTimeListWidget.ContextMenu["时点变换"]["主菜单"].addAction("转周末").triggered.connect(self.toWeekLastDay)
        self.DateTimeListWidget.ContextMenu["时点变换"]["主菜单"].addAction("转周初").triggered.connect(self.toWeekFirstDay)
        self.DateTimeListWidget.ContextMenu["时点变换"]["主菜单"].addAction("转年末").triggered.connect(self.toYearLastDay)
        self.DateTimeListWidget.ContextMenu["时点变换"]["主菜单"].addAction("转年初").triggered.connect(self.toYearFirstDay)
        self.DateTimeListWidget.ContextMenu["时点变换"]["主菜单"].addAction("转季末").triggered.connect(self.toQuarterLastDay)
        self.DateTimeListWidget.ContextMenu["时点变换"]["主菜单"].addAction("转季初").triggered.connect(self.toQuarterFirstDay)
        self.DateTimeListWidget.ContextMenu["时点变换"]["主菜单"].addAction("转财报季末").triggered.connect(self.toFinancialQuarterLastDay)
        self.DateTimeListWidget.ContextMenu["时点变换"]["主菜单"].addAction("转财报季初").triggered.connect(self.toFinancialQuarterFirstDay)
        self.DateTimeListWidget.ContextMenu["时点变换"]["主菜单"].addAction("等间隔抽样").triggered.connect(self.sampleData)
        self.DateTimeListWidget.ContextMenu["主菜单"].addMenu(self.DateTimeListWidget.ContextMenu["时点变换"]["主菜单"])
        self.DateTimeListWidget.ContextMenu["导入导出"] = {"主菜单": QMenu("导入导出", parent=self.DateTimeListWidget.ContextMenu["主菜单"])}
        self.DateTimeListWidget.ContextMenu["导入导出"]["主菜单"].addAction("导入时点").triggered.connect(self.importData)
        self.DateTimeListWidget.ContextMenu["导入导出"]["主菜单"].addAction("导出时点").triggered.connect(self.exportData)
        self.DateTimeListWidget.ContextMenu["主菜单"].addMenu(self.DateTimeListWidget.ContextMenu["导入导出"]["主菜单"])
        return 0
    def populateDateTimeListWidget(self, dts):
        self.DateTimeListWidget.clear()
        self.DateTimeListWidget.addItems([iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts])
        self.DateTimeNumEdit.setText(str(len(dts)))
        return 0
    def changeDateTime(self, dts, dt_period):
        if callable(dt_period): return dt_period(dts)
        elif dt_period=="月末日": return DateTimeFun.getMonthLastDateTime(dts)
        elif dt_period=="周末日": return DateTimeFun.getWeekLastDateTime(dts)
        elif dt_period=="年末日": return DateTimeFun.getYearLastDateTime(dts)
        elif dt_period=="季末日": return DateTimeFun.getQuarterLastDateTime(dts)
        elif dt_period=="月初日": return DateTimeFun.getMonthFirstDateTime(dts)
        elif dt_period=="周初日": return DateTimeFun.getWeekFirstDateTime(dts)
        elif dt_period=="年初日": return DateTimeFun.getYearFirstDateTime(dts)
        elif dt_period=="季初日": return DateTimeFun.getQuarterFirstDateTime(dts)
        elif dt_period=="财报季初日": return DateTimeFun.getFinancialQuarterFirstDateTime(dts)
        elif dt_period=="财报季末日": return DateTimeFun.getFinancialQuarterLastDateTime(dts)
        elif dt_period=="月中日":
            Middle, isOK = QInputDialog.getInt(self, "月中日", "月中分界日: ", value=15, min=1, max=31, step=1)
            if isOK: return DateTimeFun.getMonthMiddleDateTime(dts, middle_day=Middle)
        return dts
    @pyqtSlot()
    def on_SelectFDBDTButton_clicked(self):
        StartDT, EndDT = self.StartDTEdit.dateTime().toPyDateTime(), self.EndDTEdit.dateTime().toPyDateTime()
        TargetID = self.IDEdit.text()
        if not TargetID: TargetID = None
        DTs = self.FT.getDateTime(ifactor_name=self.FactorComboBox.currentText(), iid=TargetID, start_dt=StartDT, end_dt=EndDT)
        self.DateTimes = sorted(mergeSet(set(DTs), set(self.DateTimes), merge_type=self.FDTSelectTypeComboBox.currentText()))
        return self.populateDateTimeListWidget(self.DateTimes)
    @pyqtSlot()
    def on_SelectDateButton_clicked(self):
        StartDate, EndDate = self.StartDateEdit.date().toPyDate(), self.EndDateEdit.date().toPyDate()
        if StartDate>EndDate: Dates = []
        else:
            DateType = self.DateTypeComboBox.currentText()
            if DateType=="自然日": Dates = DateTimeFun.getDateSeries(StartDate, EndDate)
            elif DateType=="交易日":
                try:
                    Dates = self.FDB.getTradeDay(StartDate, EndDate, exchange=self.ExchangeEdit.text())
                except Exception as e:
                    return QMessageBox.critical(self, "错误", str(e))
        Dates = self.changeDateTime(Dates, self.DatePeriodComboBox.currentText())
        self.Dates = sorted(mergeSet(set(Dates), set(self.Dates), merge_type=self.DateSelectTypeComboBox.currentText()))
        return self.populateDateListWidget(self.Dates)
    @pyqtSlot()
    def on_SelectTimeButton_clicked(self):
        Times = []
        TimeDelta = self.TimePeriodEdit.time().toPyTime()
        TimeDelta = dt.timedelta(hours=TimeDelta.hour, minutes=TimeDelta.minute, seconds=TimeDelta.second)
        AMStartTime, AMEndTime = self.AMStartTimeEdit.time().toPyTime(), self.AMEndTimeEdit.time().toPyTime()
        if AMStartTime<=AMEndTime: Times += DateTimeFun.getTimeSeries(AMStartTime, AMEndTime, timedelta=TimeDelta)
        PMStartTime, PMEndTime = self.PMStartTimeEdit.time().toPyTime(), self.PMEndTimeEdit.time().toPyTime()
        if PMStartTime<=PMEndTime: Times += DateTimeFun.getTimeSeries(PMStartTime, PMEndTime, timedelta=TimeDelta)
        self.Times = sorted(mergeSet(set(Times), set(self.Times), merge_type=self.TimeSelectTypeComboBox.currentText()))
        return self.populateTimeListWidget(self.Times)
    @pyqtSlot()
    def on_SelectDTButton_clicked(self):
        if self.Times: DTs = set(DateTimeFun.combineDateTime(self.Dates, self.Times))
        else: DTs = set(DateTimeFun.combineDateTime(self.Dates, [dt.time(0)]))
        self.DateTimes = sorted(mergeSet(DTs, set(self.DateTimes), merge_type=self.DTSelectTypeComboBox.currentText()))
        return self.populateDateTimeListWidget(self.DateTimes)
    @pyqtSlot(str)
    def on_FTComboBox_currentTextChanged(self, p0):
        self.FT = self.FDB.getTable(p0)
        self.FactorComboBox.clear()
        self.FactorComboBox.addItems(self.FT.FactorNames)
    
if __name__=="__main__":
    import QuantStudio.api as QS
    FDB = QS.FactorDB.WindDB2()
    FDB.connect()
    FT = FDB.getTable("中国A股日行情")
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    TestWindow = DateTimeSetupDlg(None, dts=FT.getDateTime(start_dt=dt.datetime(2018,11,1)), ft=FT)
    TestWindow.show()
    app.exec_()
    sys.exit()
