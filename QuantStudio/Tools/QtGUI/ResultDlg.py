# -*- coding: utf-8 -*-
import os
import sys
import datetime as dt
import tempfile
from cycler import cycler

from PyQt5 import QtWidgets, QtCore, QtGui

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import matplotlib.cm
import seaborn as sns
#sns.set()
import plotly
from traits.api import File, Enum, List

from QuantStudio import __QS_MainPath__, __QS_Error__, __QS_Object__
from QuantStudio.Tools.FileFun import writeDictSeries2CSV, exportOutput2CSV, readCSV2StdDF
from QuantStudio.Tools.DataTypeFun import getNestedDictItems, getNestedDictValue, removeNestedDictItem
from QuantStudio.Tools.AuxiliaryFun import genAvailableName, joinList
from QuantStudio.Tools import StrategyTestFun
from QuantStudio.Tools.QtGUI.QtGUIFun import populateTableWithDataFrame, populateQTreeWidgetWithNestedDict
from QuantStudio.Tools.QtGUI.Ui_ResultDlg import Ui_ResultDlg

# 展示的数据类型 output: 嵌套的字典, 叶节点是 DataFrame

# 用于多选的 QTableWidget
class _MultiOptionTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None, data_range=[], data=None):
        super().__init__(parent)
        self.DataRange = data_range
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        nData = len(data_range)
        self.setRowCount(nData)
        self.setColumnCount(1)
        self.setHorizontalHeaderLabels(['参数值'])
        self.setVerticalHeaderLabels([str(iData) for iData in data_range])
        if data is None:
            for i in range(nData):
                iWidget = QtWidgets.QCheckBox(None)
                iWidget.setChecked(True)
                self.setCellWidget(i,0,iWidget)
        else:
            for i,iData in enumerate(data_range):
                iWidget = QtWidgets.QCheckBox(None)
                iWidget.setChecked((iData in data))
                self.setCellWidget(i,0,iWidget)
        # 设置弹出菜单
        self.addAction(QtWidgets.QAction('全选', self, triggered=self.selectAll))
        self.addAction(QtWidgets.QAction('全不选', self, triggered=self.selectNone))
        self.addAction(QtWidgets.QAction('反选', self, triggered=self.selectOpposite))
        return
    def extract(self):
        return [self.DataRange[i] for i in range(self.rowCount()) if self.cellWidget(i, 0).isChecked()]
    # 全选
    def selectAll(self):
        for iRow in range(self.rowCount()):
            iItem = self.cellWidget(iRow,0).setChecked(True)
    # 全不选
    def selectNone(self):
        for iRow in range(self.rowCount()):
            iItem = self.cellWidget(iRow,0).setChecked(False)
    # 反选
    def selectOpposite(self):
        for iRow in range(self.rowCount()):
            iItem = self.cellWidget(iRow,0)
            iItem.setChecked(not iItem.isChecked())



# 表格对话框
class _TableDlg(QtWidgets.QDialog):
    def __init__(self, parent=None, table_widget=None, timely_modified=False):
        super().__init__(parent)
        self.setObjectName("_TableDlg")
        self.resize(346, 472)
        self.setSizeGripEnabled(True)
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        if table_widget is not None:
            self.MainTable = table_widget
            self.MainTable.setParent(self)
        else:
            self.MainTable = QtWidgets.QTableWidget(self)
        self.MainTable.TimelyModified = timely_modified
        self.MainTable.setObjectName("MainTable")
        self.verticalLayout.addWidget(self.MainTable)
        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)
        self.isOK = False
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("_TableDlg", "_TableDlg", None))
        QtCore.QMetaObject.connectSlotsByName(self)
        return
    @QtCore.pyqtSlot()
    def on_buttonBox_accepted(self):
        self.isOK = True
        self.close()
    @QtCore.pyqtSlot()
    def on_buttonBox_rejected(self):
        self.close()
class _FromCSVArgs(__QS_Object__):
    FilePath = File(filter=["Excel (*.csv)"], arg_type="File", order=0, label="导入文件")
    RowIndex = Enum("时间", "字符串", "整数", "小数", arg_type="SingleOption", order=1, label="行索引")
    ColIndex = Enum("字符串", "时间", "整数", "小数", arg_type="SingleOption", order=2, label="列索引")
    CharSet = Enum("自动检测", "utf-8", "ascii", "mbcs", "gb2312", "gbk", "big5", "gb18030", "cp936", arg_type="SingleOption", order=3, label="字符编码")
# 基于 plotly 绘图的 ResultDlg
class PlotlyResultDlg(QtWidgets.QDialog, Ui_ResultDlg):
    def __init__(self, parent=None, output={}):
        super().__init__(parent)
        self.setupUi(self)
        self.Output = output
        self.CurDF = pd.DataFrame()# 当前显示在Table中的数据，DataFrame
        
        self.populateMainResultTree()
        self.setMenu()
        return
    def setMenu(self):# 设置弹出菜单
        # 设置 MainResultTree 的弹出菜单
        self.MainResultTree.addAction(QtWidgets.QAction('重命名',self.MainResultTree, triggered=self.renameVar))
        self.MainResultTree.addAction(QtWidgets.QAction('删除变量',self.MainResultTree, triggered=self.deleteVar))
        self.MainResultTree.addAction(QtWidgets.QAction('导出CSV',self.MainResultTree, triggered=self.toCSV))
        self.MainResultTree.addAction(QtWidgets.QAction('导出Excel',self.MainResultTree, triggered=self.toExcel))
        self.MainResultTree.addAction(QtWidgets.QAction('导入CSV',self.MainResultTree, triggered=self.fromCSV))
        # 设置 MainResultTable 的弹出菜单
        self.MainResultTable.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.MainResultTable.customContextMenuRequested.connect(self.showMainResultTableContextMenu)
        self.MainResultTable.ContextMenu = {'主菜单':QtWidgets.QMenu()}
        self.MainResultTable.ContextMenu['数据操作'] = {"主菜单":QtWidgets.QMenu("数据操作")}
        NewAction = self.MainResultTable.ContextMenu['数据操作']['主菜单'].addAction('存为变量')
        NewAction.triggered.connect(self.saveAsVar)
        NewAction = self.MainResultTable.ContextMenu['数据操作']['主菜单'].addAction('删除列')
        NewAction.triggered.connect(self.deleteColumn)
        NewAction = self.MainResultTable.ContextMenu['数据操作']['主菜单'].addAction('作为索引')
        NewAction.triggered.connect(self.asIndex)
        NewAction = self.MainResultTable.ContextMenu['数据操作']['主菜单'].addAction('取消索引')
        NewAction.triggered.connect(self.resetIndex)
        NewAction = self.MainResultTable.ContextMenu['数据操作']['主菜单'].addAction('形成Grid')
        NewAction.triggered.connect(self.asGrid)
        self.MainResultTable.ContextMenu['主菜单'].addMenu(self.MainResultTable.ContextMenu['数据操作']['主菜单'])
        self.MainResultTable.ContextMenu['排序筛选'] = {"主菜单":QtWidgets.QMenu("排序筛选")}
        NewAction = self.MainResultTable.ContextMenu['排序筛选']['主菜单'].addAction('升序')
        NewAction.triggered.connect(self.sortDataAscending)
        NewAction = self.MainResultTable.ContextMenu['排序筛选']['主菜单'].addAction('降序')
        NewAction.triggered.connect(self.sortDataDescending)
        NewAction = self.MainResultTable.ContextMenu['排序筛选']['主菜单'].addAction('筛选')
        NewAction.triggered.connect(self.filterData)
        self.MainResultTable.ContextMenu['主菜单'].addMenu(self.MainResultTable.ContextMenu['排序筛选']['主菜单'])
        self.MainResultTable.ContextMenu['绘制图像'] = {"主菜单":QtWidgets.QMenu("绘制图像")}
        NewAction = self.MainResultTable.ContextMenu['绘制图像']['主菜单'].addAction('直方图')
        NewAction.triggered.connect(self.plotHist)
        NewAction = self.MainResultTable.ContextMenu['绘制图像']['主菜单'].addAction('三维直方图')
        NewAction.triggered.connect(self.plotHist3D)
        NewAction = self.MainResultTable.ContextMenu['绘制图像']['主菜单'].addAction('二维散点图')
        NewAction.triggered.connect(self.plotScatter)
        NewAction = self.MainResultTable.ContextMenu['绘制图像']['主菜单'].addAction('三维散点图')
        NewAction.triggered.connect(self.plotScatter3D)
        NewAction = self.MainResultTable.ContextMenu['绘制图像']['主菜单'].addAction('经验分布图')
        NewAction.triggered.connect(self.plotCDF)
        NewAction = self.MainResultTable.ContextMenu['绘制图像']['主菜单'].addAction('热图')
        NewAction.triggered.connect(self.plotHeatMap)
        self.MainResultTable.ContextMenu['主菜单'].addMenu(self.MainResultTable.ContextMenu['绘制图像']['主菜单'])
        NewAction = self.MainResultTable.ContextMenu['主菜单'].addAction('统计量')
        NewAction.triggered.connect(self.calStatistics)
        self.MainResultTable.ContextMenu['数据运算'] = {"主菜单":QtWidgets.QMenu("数据运算")}
        NewAction = self.MainResultTable.ContextMenu['数据运算']['主菜单'].addAction('累积求和')
        NewAction.triggered.connect(self.calCumsum)
        NewAction = self.MainResultTable.ContextMenu['数据运算']['主菜单'].addAction('累积求积')
        NewAction.triggered.connect(self.calCumprod)
        NewAction = self.MainResultTable.ContextMenu['数据运算']['主菜单'].addAction('滚动平均')
        NewAction.triggered.connect(self.calRollingAverage)
        NewAction = self.MainResultTable.ContextMenu['数据运算']['主菜单'].addAction('自然对数')
        NewAction.triggered.connect(self.calLog)
        NewAction = self.MainResultTable.ContextMenu['数据运算']['主菜单'].addAction('收益率')
        NewAction.triggered.connect(self.calReturn)
        NewAction = self.MainResultTable.ContextMenu['数据运算']['主菜单'].addAction('对数收益率')
        NewAction.triggered.connect(self.calLogReturn)
        NewAction = self.MainResultTable.ContextMenu['数据运算']['主菜单'].addAction('净值')
        NewAction.triggered.connect(self.calWealth)
        NewAction = self.MainResultTable.ContextMenu['数据运算']['主菜单'].addAction('一阶差分')
        NewAction.triggered.connect(self.calDiff)
        NewAction = self.MainResultTable.ContextMenu['数据运算']['主菜单'].addAction('相关系数')
        NewAction.triggered.connect(self.calCorrMatrix)
        self.MainResultTable.ContextMenu['主菜单'].addMenu(self.MainResultTable.ContextMenu['数据运算']['主菜单'])
        self.MainResultTable.ContextMenu['策略统计'] = {"主菜单":QtWidgets.QMenu("策略统计")}
        NewAction = self.MainResultTable.ContextMenu['策略统计']['主菜单'].addAction('概括统计')
        NewAction.triggered.connect(self.summaryStrategy)
        NewAction = self.MainResultTable.ContextMenu['策略统计']['主菜单'].addAction('滚动统计')
        NewAction.triggered.connect(self.summaryStrategyExpanding)
        NewAction = self.MainResultTable.ContextMenu['策略统计']['主菜单'].addAction('年度统计')
        NewAction.triggered.connect(self.summaryStrategyPerYear)
        NewAction = self.MainResultTable.ContextMenu['策略统计']['主菜单'].addAction('月度统计')
        NewAction.triggered.connect(self.summaryStrategyPerYearMonth)
        NewAction = self.MainResultTable.ContextMenu['策略统计']['主菜单'].addAction('月度平均收益')
        NewAction.triggered.connect(self.calcAvgReturnPerMonth)
        NewAction = self.MainResultTable.ContextMenu['策略统计']['主菜单'].addAction('周度日平均收益')
        NewAction.triggered.connect(self.calcAvgReturnPerWeekday)
        NewAction = self.MainResultTable.ContextMenu['策略统计']['主菜单'].addAction('月度日平均收益')
        NewAction.triggered.connect(self.calcAvgReturnPerMonthday)
        NewAction = self.MainResultTable.ContextMenu['策略统计']['主菜单'].addAction('年度日平均收益')
        NewAction.triggered.connect(self.calcAvgReturnPerYearday)
        self.MainResultTable.ContextMenu['主菜单'].addMenu(self.MainResultTable.ContextMenu['策略统计']['主菜单'])
        NewAction = self.MainResultTable.ContextMenu['主菜单'].addAction('刷新')
        NewAction.triggered.connect(self.populateMainResultTable)
    # ------------------------ MainResultTree 相关操作-------------------------
    def populateMainResultTree(self):
        self.MainResultTree.blockSignals(True)
        self.MainResultTree.clear()
        populateQTreeWidgetWithNestedDict(self.MainResultTree, self.Output)
        self.MainResultTree.resizeColumnToContents(0)
        self.MainResultTree.blockSignals(False)
        return 0
    @QtCore.pyqtSlot(QtWidgets.QTreeWidgetItem, int)
    def on_MainResultTree_itemDoubleClicked(self, item, column):
        self.on_GenTableButton_clicked()
    def renameVar(self):# 重命名变量
        SelectedItem = self.MainResultTree.selectedItems()
        if len(SelectedItem)!=1: return QtWidgets.QMessageBox.critical(self, "错误", "请选择一个变量或变量集!")
        SelectedItem = SelectedItem[0]
        OldKey = SelectedItem.text(0)
        NewKey, isOk = QtWidgets.QInputDialog.getText(self, "重命名", "请输入新名字: ", text=OldKey)
        if (not isOk) or (NewKey==OldKey): return 0
        KeyList = SelectedItem.data(0, QtCore.Qt.UserRole)
        Parent = getNestedDictValue(self.Output, KeyList[:-1])
        if NewKey in Parent: return QtWidgets.QMessageBox.critical(self, "错误", "有重名!")
        else: Parent[NewKey] = Parent.pop(OldKey)
        SelectedItem.setText(0, NewKey)
        SelectedItem.setData(0, QtCore.Qt.UserRole, KeyList[:-1]+(NewKey,))
        if isinstance(Parent[NewKey], dict):
            iItem = QtWidgets.QTreeWidgetItemIterator(SelectedItem)
            iItem = iItem.__iadd__(1)
            iTreeItem = iItem.value()
            while iTreeItem:
                iParent = iTreeItem.parent()
                iParentKeyList = iParent.data(0, QtCore.Qt.UserRole)
                iTreeItem.setData(0, QtCore.Qt.UserRole, iParentKeyList+iTreeItem.data(0, QtCore.Qt.UserRole)[-1:])
                iItem = iItem.__iadd__(1)
                iTreeItem = iItem.value()
        return 0
    def deleteVar(self):# 删除变量
        SelectedItems = self.MainResultTree.selectedItems()
        for iItem in SelectedItems:
            iKeyList = iItem.data(0, QtCore.Qt.UserRole)
            removeNestedDictItem(self.Output, iKeyList)
        self.populateMainResultTree()
        return 0
    def toCSV(self):# 导出变量到CSV
        SelectedItems = self.MainResultTree.selectedItems()
        SelectedOutput = []# [(key_list, value)]
        for iItem in SelectedItems:
            iKeyList = iItem.data(0, QtCore.Qt.UserRole)
            iValue = getNestedDictValue(self.Output, iKeyList)
            if not isinstance(iValue, dict):
                SelectedOutput.append((iKeyList, iValue))
            else:
                SelectedOutput.extend(getNestedDictItems(iValue, iKeyList))
        Output = {joinList(iKeyList,"-"):iOutput for iKeyList, iOutput in SelectedOutput}
        if Output=={}: return 0
        DirPath = QtWidgets.QFileDialog.getExistingDirectory(parent=self, caption="导出CSV", directory=os.getcwd())
        if DirPath=='': return 0
        exportOutput2CSV(Output, DirPath)
        return QtWidgets.QMessageBox.information(self, "完成", "导出数据完成!")
    def toExcel(self):# 导出变量到Excel
        SelectedItems = self.MainResultTree.selectedItems()
        SelectedOutput = []# [(key_list, value)]
        for iItem in SelectedItems:
            iKeyList = iItem.data(0, QtCore.Qt.UserRole)
            iValue = getNestedDictValue(self.Output, iKeyList)
            if not isinstance(iValue, dict):
                SelectedOutput.append((iKeyList, iValue))
            else:
                SelectedOutput.extend(getNestedDictItems(iValue, iKeyList))
        FilePath, _ = QtWidgets.QFileDialog.getSaveFileName(parent=self, caption="导出 Excel", directory=os.getcwd(), filter="Excel (*.xls)")
        if (not FilePath) or (not SelectedOutput): return 0
        Writer = pd.ExcelWriter(FilePath)
        for iKeyList, iOutput in SelectedOutput:
            iSheetName = joinList(iKeyList, "-")
            iOutput.to_excel(Writer, sheet_name=iSheetName, header=True, index=True, engine="xlwt")
        Writer.save()
        return QtWidgets.QMessageBox.information(self, "完成", "导出数据完成!")
    def fromCSV(self):# 将CSV数据导入变量, TODO
        Args = _FromCSVArgs()
        if (not Args.setArgs()) or (not Args["导入文件"]): return 0
        if isinstance(Args["导入文件"], str): Files = [Args["导入文件"]]
        else: Files = Args["导入文件"]
        SelectedItem = self.MainResultTree.selectedItems()
        if SelectedItem==[]: Parent = self.MainResultTree
        elif SelectedItem[0].childCount()==0: Parent = SelectedItem[0].parent()
        else: Parent = SelectedItem[0]
        if (Parent is None) or (Parent is self.MainResultTree):
            ParentOutput = self.Output
            ParentKeyList = ()
        else:
            ParentKeyList = Parent.data(0, QtCore.Qt.UserRole)
            ParentOutput = getNestedDictValue(self.Output, ParentKeyList)
        for iFilePath in Files:
            iVar = os.path.split(iFilePath)[-1][:-4]
            if (iVar in ParentOutput) and (QtWidgets.QMessageBox.Ok!=QtWidgets.QMessageBox.question(self, "警告", "变量: "+iVar+", 重名, 是否覆盖?", QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Cancel)):
                iVar = genAvailableName(iVar, list(ParentOutput.keys()))
            iData = readCSV2StdDF(iFilePath,index=Args["行索引"],col=Args["列索引"],encoding=(None if Args["字符编码"]=="自动检测" else Args["字符编码"]))
            ParentOutput[iVar] = iData
        self.populateMainResultTree()
        return QtWidgets.QMessageBox.information(self, "完成", "导入数据完成!")
    # ------------------------MainResultTable 相关操作-------------------------
    def populateMainResultTable(self):# 刷新数据
        TargetDF = self.CurDF
        if self.RowLimitCheckBox.isChecked():
            RowLimitNum = self.RowLimitSpinBox.value()
            TargetDF = TargetDF.iloc[:min((RowLimitNum,TargetDF.shape[0]))]
        if self.ColumnLimitCheckBox.isChecked():
            ColLimitNum = self.ColumnLimitSpinBox.value()
            TargetDF = TargetDF.iloc[:,:min((ColLimitNum,TargetDF.shape[1]))]
        self.RowColLabel.setText("%d 行, %d 列" % self.CurDF.shape)
        return populateTableWithDataFrame(self.MainResultTable,TargetDF)    
    def showMainResultTableContextMenu(self,pos):# 显示MainResultTable的右键菜单
        self.MainResultTable.ContextMenu['主菜单'].move(QtGui.QCursor.pos())
        self.MainResultTable.ContextMenu['主菜单'].show()
    # -----------------------------------数据操作--------------------------------
    def saveDF(self, df, var_name=""):
        VarName, isOk = QtWidgets.QInputDialog.getText(self, "变量名称", "请输入变量名称: ", text=var_name)
        if (not isOk) or (VarName==""): return 0
        SelectedItem = self.MainResultTree.selectedItems()
        if SelectedItem==[]:
            Parent = self.MainResultTree
        elif SelectedItem[0].childCount()==0:
            Parent = SelectedItem[0].parent()
        else:
            Parent = SelectedItem[0]
        if (Parent is None) or (Parent is self.MainResultTree):
            ParentOutput = self.Output
            ParentKeyList = ()
        else:
            ParentKeyList = Parent.data(0, QtCore.Qt.UserRole)
            ParentOutput = getNestedDictValue(self.Output, ParentKeyList)
        if (VarName in ParentOutput) and (QtWidgets.QMessageBox.Ok!=QtWidgets.QMessageBox.question(self, "警告", "变量: "+VarName+", 重名, 是否覆盖?", QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Cancel)):
            return 0
        ParentOutput[VarName] = df
        return self.populateMainResultTree()
    def deleteColumn(self):# 删除列
        SelectedColumns = self.getSelectedColumns()
        SelectedColumns = [self.CurDF.columns[iCol] for iCol in SelectedColumns]
        for iCol in SelectedColumns:
            del self.CurDF[iCol]
        return self.populateMainResultTable()
    def saveAsVar(self):# 存为变量
        return self.saveDF(self.CurDF)
    def asIndex(self):# 作为索引
        SelectedColumn = self.getSelectedColumns()
        if len(SelectedColumn)!=1: return QtWidgets.QMessageBox.critical(self, "错误", "请选择一列!")
        self.CurDF = self.CurDF.reset_index().set_index([self.CurDF.columns[SelectedColumn[0]]])
        return self.populateMainResultTable()
    def resetIndex(self):# 取消索引
        self.CurDF = self.CurDF.reset_index()
        return self.populateMainResultTable()
    def asGrid(self):# 形成Grid
        SelectedColumns = self.getSelectedColumns()
        if len(SelectedColumns)!=3: return QtWidgets.QMessageBox.critical(self, "错误", "请选择三列!")
        X = self.CurDF.iloc[:,SelectedColumns[0]]
        Y = self.CurDF.iloc[:,SelectedColumns[1]]
        Z = self.CurDF.iloc[:,SelectedColumns[2]]
        XValues = X.unique()
        YValues = Y.unique()
        XValues.sort()
        YValues.sort()
        self.CurDF = pd.DataFrame(index=XValues,columns=YValues)
        for iX in XValues:
            for jY in YValues:
                ijMask = ((X==iX) & (Y==jY))
                self.CurDF.loc[iX,jY] = Z[ijMask].iloc[0]
        return self.populateMainResultTable()
    # -----------------------------------排序筛选--------------------------------
    def filterData(self):# 筛选
        SelectedColumn = self.getSelectedColumns()
        if len(SelectedColumn)!=1: return QtWidgets.QMessageBox.critical(self, "错误", "请选择一列!")
        SelectedDF,Msg = self.getSelectedDF(all_num=False)
        SelectedDF = SelectedDF.iloc[:,0]
        Dlg = _TableDlg(None, _MultiOptionTable(None, list(SelectedDF.unique()), None))
        Dlg.exec_()
        if not Dlg.isOK: return 0
        SelectedData = set(Dlg.MainTable.extract())
        Mask = pd.Series(False,index=SelectedDF.index)
        for i in range(SelectedDF.shape[0]):
            if SelectedDF.iloc[i] in SelectedData: Mask.iloc[i] = True
        self.CurDF = self.CurDF[Mask]
        return self.populateMainResultTable()
    def sortData(self,ascending=True):
        SelectedColumns = self.getSelectedColumns()
        if len(SelectedColumns)==0: return QtWidgets.QMessageBox.critical(self, "错误", "请选择至少一列!")
        self.CurDF = self.CurDF.sort_values(by=list(self.CurDF.columns[SelectedColumns]), ascending=ascending)
        return self.populateMainResultTable()
    def sortDataAscending(self):# 升序
        return self.sortData()
    def sortDataDescending(self):# 降序
        return self.sortData(False)
    # -----------------------------------绘制图像--------------------------------
    def plotHist(self):# 直方图
        SelectedColumn = self.getSelectedColumns()
        if len(SelectedColumn)!=1: return QtWidgets.QMessageBox.critical(self, "错误", "请选择一列!")
        SelectedDF,Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        SelectedDF = SelectedDF.iloc[:,0]
        GroupNum, isOK = QtWidgets.QInputDialog.getInt(self, '获取分组数', '分组数', value=10, min=1, max=1000, step=1)
        if not isOK: return 0
        yData = SelectedDF[pd.notnull(SelectedDF)].values
        xData = np.linspace(np.nanmin(yData),np.nanmax(yData),yData.shape[0]*10)
        yNormalData = stats.norm.pdf(xData,loc=np.nanmean(yData),scale=np.nanstd(yData))
        GraphObj = [plotly.graph_objs.Histogram(x=yData,histnorm='probability',name='直方图',nbinsx=GroupNum),plotly.graph_objs.Scatter(x=xData,y=yNormalData,name='Normal Distribution',line={'color':'rgb(255,0,0)','width':2})]
        with tempfile.TemporaryFile() as File:
            plotly.offline.plot({"data":GraphObj,"layout": plotly.graph_objs.Layout(title="直方图")}, filename=File.name)
        return 0
    def plotHist3D(self):# 三维直方图
        from mpl_toolkits.mplot3d import Axes3D
        SelectedColumn = self.getSelectedColumns()
        if len(SelectedColumn)!=2: return QtWidgets.QMessageBox.critical(self, "错误", "请选择两列!")
        SelectedDF,Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        SelectedDF.dropna()
        xData = SelectedDF.iloc[:,0].astype('float').values
        yData = SelectedDF.iloc[:,1].astype('float').values
        GroupNum, isOK = QtWidgets.QInputDialog.getInt(self, "获取分组数", "分组数", value=10, min=1, max=1000, step=1)
        if not isOK: return 0
        hist, xedges, yedges = np.histogram2d(xData, yData, bins=GroupNum)
        elements = (len(xedges) - 1) * (len(yedges) - 1)
        xpos, ypos = np.meshgrid(xedges[:-1]+0.25, yedges[:-1]+0.25)
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros(elements)
        dx = 0.5 * np.ones_like(zpos)
        dy = dx.copy()
        dz = hist.flatten()
        tempFigDlg = _MatplotlibWidget()
        Fig = tempFigDlg.Mpl.Fig
        Axes = Axes3D(Fig)
        Axes.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
        tempFigDlg.Mpl.draw()
        tempFigDlg.show()
        return 0    
    def plotCDF(self):# 经验分布图
        SelectedColumn = self.getSelectedColumns()
        if len(SelectedColumn)!=1: return QtWidgets.QMessageBox.critical(self, "错误", "请选择一列!")
        SelectedDF,Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        SelectedDF = SelectedDF.iloc[:,0]
        xData = SelectedDF[pd.notnull(SelectedDF)].values
        xData.sort()
        nData = xData.shape[0]
        Delta = (xData[-1]-xData[0])/nData
        xData = np.append(xData[0]-Delta,xData)
        xData = np.append(xData,xData[-1]+Delta)
        yData = (np.linspace(0,nData+1,nData+2))/(nData)
        yData[-1] = yData[-2]
        GraphObj = [plotly.graph_objs.Scatter(x=xData,y=yData,name="经验分布函数")]
        xNormalData = np.linspace(xData[0],xData[-1],(nData+2)*10)
        yNormalData = stats.norm.cdf(xNormalData,loc=np.mean(xData[1:-1]),scale=np.std(xData[1:-1]))
        GraphObj.append(plotly.graph_objs.Scatter(x=xNormalData,y=yNormalData,name="Normal Distribution"))
        with tempfile.TemporaryFile() as File:
            plotly.offline.plot({"data":GraphObj,"layout": plotly.graph_objs.Layout(title="经验分布")}, filename=File.name)
        return 0
    def plotScatter(self):# 二维散点图
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        elif (SelectedDF.shape[1]<1) or (SelectedDF.shape[1]>3): return QtWidgets.QMessageBox.critical(self, "错误", "请选择一到三列!")
        isOK = QtWidgets.QMessageBox.question(self, "添加回归线", "是否添加回归线?", QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Cancel)
        SelectedDF.dropna()
        GraphObj = []
        if SelectedDF.shape[1]==1:
            xData = np.linspace(0,SelectedDF.shape[0]-1,SelectedDF.shape[0])
            yData = SelectedDF.iloc[:,0].values
            GraphObj.append(plotly.graph_objs.Scatter(x=xData,y=yData,mode="markers",name=str(SelectedDF.columns[0])))
        if SelectedDF.shape[1]==2:
            xData = SelectedDF.iloc[:,0].values
            yData = SelectedDF.iloc[:,1].values
            GraphObj.append(plotly.graph_objs.Scatter(x=xData,y=yData,mode="markers",name=str(SelectedDF.columns[0])+"-"+str(SelectedDF.columns[1])))
        elif SelectedDF.shape[1]==3:
            xData = SelectedDF.iloc[:,0].values
            yData = SelectedDF.iloc[:,1].values
            zData = SelectedDF.iloc[:,2].astype('float')
            Size = ((zData-zData.mean())/zData.std()*50).values
            GraphObj.append(plotly.graph_objs.Scatter(x=xData,y=yData,marker=dict(size=Size),mode="markers",name=str(SelectedDF.columns[0])+"-"+str(SelectedDF.columns[1])+"-"+str(SelectedDF.columns[2])))
        if isOK==QtWidgets.QMessageBox.Ok:
            xData = sm.add_constant(xData, prepend=True)
            Model = sm.OLS(yData,xData,missing='drop')
            Result = Model.fit()
            xData = xData[:,1]
            xData.sort()
            yRegressData = Result.params[0]+Result.params[1]*xData
            GraphObj.append(plotly.graph_objs.Scatter(x=xData,y=yRegressData,name="回归线"))
        with tempfile.TemporaryFile() as File:
            plotly.offline.plot({"data":GraphObj,"layout": plotly.graph_objs.Layout(title="散点图")}, filename=File.name)
        return 0
    def plotScatter3D(self):# 三维散点图
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        elif SelectedDF.shape[1]!=3: return QtWidgets.QMessageBox.critical(self, "错误", "请选择三列!")
        isOK = QtWidgets.QMessageBox.question(self, "添加回归面", "是否添加回归面?", QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Cancel)
        SelectedDF.dropna()
        xData = SelectedDF.iloc[:,0].values
        yData = SelectedDF.iloc[:,1].values
        zData = SelectedDF.iloc[:,2].values
        GraphObj = [plotly.graph_objs.Scatter3d(x=xData,y=yData,z=zData,mode='markers',name=str(SelectedDF.columns[0])+"-"+str(SelectedDF.columns[1])+"-"+str(SelectedDF.columns[2]))]
        if isOK==QtWidgets.QMessageBox.Ok:
            xRegressData = np.ones((SelectedDF.shape[0],3))
            xRegressData[:,1] = xData
            xRegressData[:,2] = yData
            Model = sm.OLS(zData,xRegressData,missing='drop')
            Result = Model.fit()
            xData.sort()
            yData.sort()
            X,Y = np.meshgrid(xData,yData)
            zRegressData = Result.params[0]+Result.params[1]*X+Result.params[2]*Y
            GraphObj.append(plotly.graph_objs.Surface(x=X,y=Y,z=zRegressData,colorscale='Viridis',name='回归面'))
        with tempfile.TemporaryFile() as File:
            plotly.offline.plot({"data":GraphObj,"layout": plotly.graph_objs.Layout(title="3D散点图")}, filename=File.name)
        return 0
    def plotHeatMap(self):# 热图
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        Dlg = _TableDlg(None, _MultiOptionTable(None, SelectedDF.index.tolist(), None))
        Dlg.exec_()
        if not Dlg.isOK: return 0
        SelectedIndex = Dlg.MainTable.extract()
        SelectedDF = SelectedDF.loc[SelectedIndex]
        SelectedIndex = [str(iIndex) for iIndex in SelectedIndex]
        GraphObj = [plotly.graph_objs.Heatmap(z=SelectedDF.astype('float').values)]
        with tempfile.TemporaryFile() as File:
            plotly.offline.plot({"data":GraphObj,"layout": plotly.graph_objs.Layout(title="热图")},filename=File.name)
        return 0
    # -----------------------------------数据运算--------------------------------
    def calStatistics(self):# 统计量
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        # 设置要统计的索引
        SelectedIndex = self._getDataIndex(list(SelectedDF.index))
        SummaryData = pd.DataFrame(index=['数量','均值','中位数','方差','标准差','最大值','最小值','总和','总积'],columns=[str(iCol) for iCol in SelectedDF.columns])
        for i,iCol in enumerate(SelectedDF.columns):
            iData = SelectedDF.iloc[:,i].loc[SelectedIndex]
            SummaryData.loc['总和'].iloc[i] = iData.sum()
            SummaryData.loc['数量'].iloc[i] = iData[pd.notnull(iData)].shape[0]
            SummaryData.loc['均值'].iloc[i] = iData.mean()
            SummaryData.loc['方差'].iloc[i] = iData.var()
            SummaryData.loc['标准差'].iloc[i] = iData.std()
            SummaryData.loc['中位数'].iloc[i] = iData.median()
            SummaryData.loc['总积'].iloc[i] = iData.prod()
            SummaryData.loc['最大值'].iloc[i] = iData.max()
            SummaryData.loc['最小值'].iloc[i] = iData.min()
        TableWidget = QtWidgets.QTableWidget()
        populateTableWithDataFrame(TableWidget, SummaryData)
        _TableDlg(None, TableWidget).exec_()
        return 0
    def calCumsum(self):# 累计求和
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        return self.saveDF(SelectedDF.cumsum())
    def calCumprod(self):# 累计求积
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        return self.saveDF(SelectedDF.cumprod())
    def calRollingAverage(self):# 滚动平均
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        Window,isOK = QtWidgets.QInputDialog.getInt(self, "获取窗口长度", "窗口长度", value=12, min=1, max=SelectedDF.shape[0], step=1)
        if not isOK: return 0
        return self.saveDF(SelectedDF.rolling(Window).mean())
    def calLog(self):# 求对数
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        return self.saveDF(np.log(SelectedDF))
    def calReturn(self):# 求收益率序列
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        return self.saveDF(pd.DataFrame(StrategyTestFun.calcYieldSeq(SelectedDF.values),index=SelectedDF.index,columns=SelectedDF.columns))
    def calLogReturn(self):# 求对数收益率序列
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        return self.saveDF(np.log(SelectedDF).diff())
    def calWealth(self):# 求净值
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        return self.saveDF(pd.DataFrame(StrategyTestFun.calcWealthSeq(SelectedDF.values),index=SelectedDF.index,columns=SelectedDF.columns))
    def calDiff(self):# 差分
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        return self.saveDF(SelectedDF.diff())
    def calCorrMatrix(self):# 相关系数
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        return self.saveDF(SelectedDF.corr())
    # -----------------------------------策略统计--------------------------------
    def _getDataIndex(self, all_index, index_type=None):
        Mask = pd.Series(True, index=all_index)
        if index_type is not None:
            for i in range(Mask.shape[0]):
                if not isinstance(all_index[i], (dt.date, dt.datetime)):
                    Mask.iloc[i] = False
        Dlg = _TableDlg(None, _MultiOptionTable(None, Mask[Mask].index.tolist(), None))
        Dlg.exec_()
        return Dlg.MainTable.extract()
    def summaryStrategy(self):
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        # 设置要统计的索引
        SelectedIndex = self._getDataIndex(SelectedDF.index.tolist(), index_type="Date")
        SelectedDF = SelectedDF.loc[SelectedIndex, :]
        if SelectedDF.shape[0]==0: return 0
        Summary = StrategyTestFun.summaryStrategy(SelectedDF.values, SelectedDF.index.tolist())
        Summary.columns = SelectedDF.columns
        return self.saveDF(Summary, var_name="统计数据")
    def summaryStrategyExpanding(self):
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        MinPeriod,isOK = QtWidgets.QInputDialog.getInt(None,'获取','最小窗口',value=20,min=1,max=10000,step=1)
        if not isOK: return 0
        # 设置要统计的索引
        SelectedIndex = self._getDataIndex(SelectedDF.index.tolist(), index_type="Date")
        SelectedDF = SelectedDF.loc[SelectedIndex,:]
        NumPerYear = (SelectedDF.shape[0]-1)/((SelectedDF.index[-1]-SelectedDF.index[0]).days/365)
        AnnualYield = pd.DataFrame(StrategyTestFun.calcExpandingAnnualYieldSeq(SelectedDF.values,MinPeriod,NumPerYear),index=SelectedDF.index,columns=[iCol+"-年化收益率" for iCol in SelectedDF.columns])
        AnnualVolatility = pd.DataFrame(StrategyTestFun.calcExpandingAnnualVolatilitySeq(SelectedDF.values,MinPeriod,NumPerYear),index=SelectedDF.index,columns=[iCol+"-年化波动率" for iCol in SelectedDF.columns])
        Sharpe = pd.DataFrame(AnnualYield.values/AnnualVolatility.values,index=SelectedDF.index,columns=[iCol+"-Sharpen比率" for iCol in SelectedDF.columns])
        return self.saveDF(pd.merge(pd.merge(AnnualYield,AnnualVolatility,left_index=True,right_index=True),Sharpe,left_index=True,right_index=True),var_name="滚动统计")
    def summaryStrategyPerYear(self):
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        # 设置要统计的索引
        SelectedIndex = self._getDataIndex(SelectedDF.index.tolist(), index_type="Date")
        SelectedDF = SelectedDF.loc[SelectedIndex, :]
        if SelectedDF.shape[0]==0: return 0
        DTs = SelectedDF.index.tolist()
        Return = StrategyTestFun.calcReturnPerYear(SelectedDF.values, DTs, dt_ruler=None)
        Return.columns = [str(iCol)+"-收益率" for iCol in SelectedDF.columns]
        Volatility = StrategyTestFun.calcVolatilityPerYear(SelectedDF.values, DTs, dt_ruler=None)
        Volatility.columns = [str(iCol)+"-波动率" for iCol in SelectedDF.columns]
        Sharpe = pd.DataFrame(Return.values/Volatility.values,index=Return.index,columns=[str(iCol)+"-Sharpe比率" for iCol in SelectedDF.columns])
        Drawdown = StrategyTestFun.calcMaxDrawdownPerYear(SelectedDF.values, DTs, dt_ruler=None)
        Drawdown.columns = [str(iCol)+"-最大回撤率" for iCol in SelectedDF.columns]
        return self.saveDF(pd.merge(pd.merge(pd.merge(Return,Volatility,left_index=True,right_index=True),Sharpe,left_index=True,right_index=True),Drawdown,left_index=True,right_index=True),var_name="年度统计")
    def summaryStrategyPerYearMonth(self):
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        # 设置要统计的索引
        SelectedIndex = self._getDataIndex(SelectedDF.index.tolist(), index_type="Date")
        SelectedDF = SelectedDF.loc[SelectedIndex,:]
        if SelectedDF.shape[0]==0: return 0
        DTs = SelectedDF.index.tolist()
        Return = StrategyTestFun.calcReturnPerYearMonth(SelectedDF.values, DTs, dt_ruler=None)
        Return.columns = [str(iCol)+"-收益率" for iCol in SelectedDF.columns]
        Volatility = StrategyTestFun.calcVolatilityPerYearMonth(SelectedDF.values, DTs, dt_ruler=None)
        Volatility.columns = [str(iCol)+"-波动率" for iCol in SelectedDF.columns]
        Sharpe = pd.DataFrame(Return.values/Volatility.values,index=Return.index,columns=[str(iCol)+"-Sharpe比率" for iCol in SelectedDF.columns])
        Drawdown = StrategyTestFun.calcMaxDrawdownPerYearMonth(SelectedDF.values, DTs, dt_ruler=None)
        Drawdown.columns = [str(iCol)+"-最大回撤率" for iCol in SelectedDF.columns]
        return self.saveDF(pd.merge(pd.merge(pd.merge(Return,Volatility,left_index=True,right_index=True),Sharpe,left_index=True,right_index=True),Drawdown,left_index=True,right_index=True),var_name="月度统计")
    def calcAvgReturnPerMonth(self):
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        # 设置要统计的索引
        SelectedIndex = self._getDataIndex(SelectedDF.index.tolist(), index_type="Date")
        SelectedDF = SelectedDF.loc[SelectedIndex,:]
        if SelectedDF.shape[0]==0: return 0
        Return = StrategyTestFun.calcAvgReturnPerMonth(SelectedDF.values, SelectedDF.index.tolist(), dt_ruler=None)
        Return.columns = SelectedDF.columns
        return self.saveDF(Return, var_name="月度平均收益率")
    def calcAvgReturnPerWeekday(self):
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        # 设置要统计的索引
        SelectedIndex = self._getDataIndex(SelectedDF.index.tolist(), index_type="Date")
        SelectedDF = SelectedDF.loc[SelectedIndex,:]
        if SelectedDF.shape[0]==0: return 0
        Return = StrategyTestFun.calcAvgReturnPerWeekday(SelectedDF.values, SelectedDF.index.tolist(), dt_ruler=None)
        Return.columns = SelectedDF.columns
        return self.saveDF(Return,var_name="周度日平均收益率")
    def calcAvgReturnPerMonthday(self):
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        # 设置要统计的索引
        SelectedIndex = self._getDataIndex(SelectedDF.index.tolist(),index_type="Date")
        SelectedDF = SelectedDF.loc[SelectedIndex,:]
        if SelectedDF.shape[0]==0: return 0
        Return = StrategyTestFun.calcAvgReturnPerMonthday(SelectedDF.values, SelectedDF.index.tolist(), dt_ruler=None)
        Return.columns = SelectedDF.columns
        return self.saveDF(Return,var_name="月度日平均收益率")
    def calcAvgReturnPerYearday(self):
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        # 设置要统计的索引
        SelectedIndex = self._getDataIndex(SelectedDF.index.tolist(), index_type="Date")
        SelectedDF = SelectedDF.loc[SelectedIndex, :]
        if SelectedDF.shape[0]==0: return 0
        Return = StrategyTestFun.calcAvgReturnPerYearday(SelectedDF.values, SelectedDF.index.tolist(), dt_ruler=None)
        Return.columns = SelectedDF.columns
        return self.saveDF(Return,var_name="年度日平均收益率")
    # -----------------------------------面板操作----------------------------------
    def getSelectedColumns(self):
        SelectedIndexes = self.MainResultTable.selectedIndexes()
        nRow = self.MainResultTable.rowCount()
        if nRow==0: return []
        SelectedColumns = []
        Rngs = self.MainResultTable.selectedRanges()
        for iRng in Rngs: SelectedColumns.extend(np.arange(iRng.leftColumn(), iRng.rightColumn()+1))
        return sorted(SelectedColumns)
    def getSelectedDF(self, all_num=True):
        SelectedColumns = self.getSelectedColumns()
        if SelectedColumns==[]: return (None, "没有选中数据列!")
        SelectedDF = self.CurDF.iloc[:, SelectedColumns].copy()
        if all_num:
            try:
                SelectedDF = SelectedDF.astype("float")
            except:
                return (None, "选择的数据中包含非数值型数据!")
        return (SelectedDF, "")
    @QtCore.pyqtSlot()
    def on_GenTableButton_clicked(self):
        SelectedItems = self.MainResultTree.selectedItems()
        SelectedOutput = []# [(key_list, value)]
        for iItem in SelectedItems:
            iKeyList = iItem.data(0, QtCore.Qt.UserRole)
            iValue = getNestedDictValue(self.Output, iKeyList)
            if not isinstance(iValue, dict):
                SelectedOutput.append((iKeyList, iValue))
            else:
                SelectedOutput.extend(getNestedDictItems(iValue, iKeyList))
        nOutput = len(SelectedOutput)
        if nOutput==0: return 0
        elif nOutput==1:
            self.CurDF = SelectedOutput[0][1]
            self.CurDF.Name = SelectedOutput[0][0][-1]
            return self.populateMainResultTable()
        MergeHow, isOK = QtWidgets.QInputDialog.getItem(self, "多表连接方式", "请选择连接方式", ['inner','outer','left','right'])
        if not isOK: return 0
        self.CurDF = SelectedOutput[0][1].copy()
        iPrefix = joinList(SelectedOutput[0][0],"-")
        self.CurDF.columns = [iPrefix+"-"+str(iCol) for iCol in self.CurDF.columns]
        for iKeyList, iOutput in SelectedOutput[1:]:
            iOutput = iOutput.copy()
            iPrefix = joinList(iKeyList,"-")
            iOutput.columns = [iPrefix+"-"+str(iCol) for iCol in iOutput.columns]
            self.CurDF = pd.merge(self.CurDF, iOutput, left_index=True, right_index=True, how=MergeHow)
        if self.CurDF.shape[0]==0: QtWidgets.QMessageBox.critical(self, "错误", "你选择的结果集索引可能不一致!")
        return self.populateMainResultTable()
    def _getPlotArgs(self, plot_data):
        nCol = plot_data.shape[1]
        PlotMode = ['Line']*nCol
        PlotAxes = ['左轴']*nCol
        DataTable = QtWidgets.QTableWidget()
        DataTable.setRowCount(nCol)
        DataTable.setVerticalHeaderLabels([str(iCol) for iCol in plot_data.columns])
        DataTable.setColumnCount(2)
        DataTable.setHorizontalHeaderLabels(['图像模式','坐标轴'])
        AllModes = ['Line','Bar','Stack']
        AllAxes = ['左轴','右轴']
        for i in range(nCol):
            iComboBox = QtWidgets.QComboBox(None)
            iComboBox.addItems(AllModes)
            DataTable.setCellWidget(i,0,iComboBox)
            iComboBox = QtWidgets.QComboBox(None)
            iComboBox.addItems(AllAxes)
            DataTable.setCellWidget(i,1,iComboBox)
        Dlg = _TableDlg(None, DataTable)
        Dlg.exec_()
        if not Dlg.isOK: return (None, None)
        for i in range(nCol):
            PlotMode[i] = DataTable.cellWidget(i,0).currentText()
            PlotAxes[i] = DataTable.cellWidget(i,1).currentText()
        return (PlotMode, PlotAxes)
    @QtCore.pyqtSlot()
    def on_PlotButton_clicked(self):
        # 获取绘图数据
        PlotResult, Msg = self.getSelectedDF(all_num=True)
        if PlotResult is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        # 设置绘图模式
        PlotMode, PlotAxes = self._getPlotArgs(PlotResult)
        if PlotMode is None: return 0
        # 设置要绘制的索引
        xData = self._getDataIndex(PlotResult.index)
        if not xData: return QtWidgets.QMessageBox.critical(self, "错误", "绘图数据为空!")
        PlotResult = PlotResult.loc[xData]
        xTickLabels = []
        isStr = False
        for iData in xData:
            xTickLabels.append(str(iData))
            if isinstance(iData, str): isStr = True
        if isStr: xData = xTickLabels
        LayoutDict = {"title":','.join([str(iCol) for iCol in PlotResult.columns])}
        if ('左轴' in PlotAxes): LayoutDict['yaxis'] = dict(title='Left Axis')
        if ('右轴' in PlotAxes): LayoutDict['yaxis2'] = dict(title='Right Axis', titlefont=dict(color='rgb(148, 103, 189)'), tickfont=dict(color='rgb(148, 103, 189)'), overlaying='y', side='right')
        GraphObj = []
        for i in range(PlotResult.shape[1]):
            iArgs = ({} if PlotAxes[i]=="左轴" else {"yaxis":"y2"})
            yData = PlotResult.iloc[:,i]
            if PlotMode[i]=='Line':
                iGraphObj = plotly.graph_objs.Scatter(x=xData, y=yData.values, name=str(yData.name), **iArgs)
            elif PlotMode[i]=='Bar':
                iGraphObj = plotly.graph_objs.Bar(x=xData, y=yData.values, name=str(yData.name), **iArgs)
            elif PlotMode[i]=='Stack':
                iGraphObj = plotly.graph_objs.Scatter(x=xData, y=yData.values, name=str(yData.name), fill='tonexty', **iArgs)
            GraphObj.append(iGraphObj)
        Fig = plotly.graph_objs.Figure(data=GraphObj, layout=plotly.graph_objs.Layout(**LayoutDict))
        with tempfile.TemporaryFile() as File:
            plotly.offline.plot(Fig, filename=File.name)
        return 0
    @QtCore.pyqtSlot()
    def on_ExportButton_clicked(self):
        if self.CurDF is None: return 0
        FileName = getattr(self.CurDF, "Name", "untitled")
        FilePath = QtWidgets.QFileDialog.getSaveFileName(self, "导出数据", ".."+os.sep+FileName+".csv", "Excel (*.csv)")
        if isinstance(FilePath, tuple): FilePath = FilePath[0]
        if not FilePath: return 0
        self.CurDF.to_csv(FilePath)
        return QtWidgets.QMessageBox.information(self, "完成", "导出数据完成!")
    @QtCore.pyqtSlot()
    def on_TransposeButton_clicked(self):
        if self.CurDF is not None:
            self.CurDF = self.CurDF.T
            return self.populateMainResultTable()
        return 0
# 基于 matplotlib 绘图的 ResultDlg
class _MplCanvas(FigureCanvas):
    def __init__(self, parent=None, fig=None, width=5, height=4, dpi=150):
        if fig is None:
            self.Fig = Figure(figsize=(width, height), dpi=dpi)
        else:
            self.Fig = fig
        #self.Axes = self.Fig.add_subplot(111)
        #self.Axes.hold(False)
        FigureCanvas.__init__(self, self.Fig)
        FigureCanvas.setSizePolicy(self,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
class _MatplotlibWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, fig=None):
        super(_MatplotlibWidget, self).__init__(parent)
        self.Layout = QtWidgets.QVBoxLayout(self)
        self.Mpl = _MplCanvas(self, fig=fig)
        self.MplNTB = NavigationToolbar(self.Mpl, self)
        self.Layout.addWidget(self.Mpl)
        self.Layout.addWidget(self.MplNTB)
        self.setLayout(self.Layout)
        
class MatplotlibResultDlg(PlotlyResultDlg):
    def setMenu(self):# 设置弹出菜单
        super().setMenu()
        NewAction = self.MainResultTable.ContextMenu['绘制图像']['主菜单'].addAction('相关图')
        NewAction.triggered.connect(self.plotCorr)
        NewAction = self.MainResultTable.ContextMenu['绘制图像']['主菜单'].addAction('联合图')
        NewAction.triggered.connect(self.plotJoint)
        NewAction = self.MainResultTable.ContextMenu['绘制图像']['主菜单'].addAction('QQ图')
        NewAction.triggered.connect(self.plotQQ)
        NewAction = self.MainResultTable.ContextMenu['绘制图像']['主菜单'].addAction('雷达图')
        NewAction.triggered.connect(self.plotRadar)
    def plotHist(self):
        SelectedColumn = self.getSelectedColumns()
        if len(SelectedColumn)!=1: return QtWidgets.QMessageBox.critical(self, "错误", "请选择一列!")
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        SelectedDF = SelectedDF.iloc[:,0]
        GroupNum,isOK = QtWidgets.QInputDialog.getInt(self, "获取分组数", "分组数", value=10, min=1, max=1000, step=1)
        if not isOK: return 0
        tempFigDlg = _MatplotlibWidget()
        Fig = tempFigDlg.Mpl.Fig
        Axes = Fig.add_subplot(111)
        yData = SelectedDF[pd.notnull(SelectedDF)].values
        xData = np.linspace(np.min(yData),np.max(yData),len(yData)*10)
        yNormalData = stats.norm.pdf(xData,loc=np.mean(yData),scale=np.std(yData))
        Axes.hist(yData, GroupNum, density=True, label='直方图', color="b")
        Axes.plot(xData, yNormalData, label='Normal Distribution', linewidth=2, color='r')
        Axes.legend(loc='upper left', shadow=True)
        tempFigDlg.Mpl.draw()
        tempFigDlg.show()
        return 0
    def plotHist3D(self):
        from mpl_toolkits.mplot3d import Axes3D
        SelectedColumn = self.getSelectedColumns()
        if len(SelectedColumn)!=2: return QtWidgets.QMessageBox.critical(self, "错误", "请选择两列!")
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        SelectedDF.dropna()
        xData = SelectedDF.iloc[:,0].astype('float').values
        yData = SelectedDF.iloc[:,1].astype('float').values
        GroupNum,isOK = QtWidgets.QInputDialog.getInt(self, "获取分组数", "分组数", value=10, min=1, max=1000, step=1)
        if not isOK: return 0
        hist, xedges, yedges = np.histogram2d(xData, yData, bins=GroupNum)
        elements = (len(xedges) - 1) * (len(yedges) - 1)
        xpos, ypos = np.meshgrid(xedges[:-1]+0.25, yedges[:-1]+0.25)
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros(elements)
        dx = 0.5 * np.ones_like(zpos)
        dy = dx.copy()
        dz = hist.flatten()
        tempFigDlg = _MatplotlibWidget()
        Fig = tempFigDlg.Mpl.Fig
        Axes = Axes3D(Fig)
        Axes.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
        tempFigDlg.Mpl.draw()
        tempFigDlg.show()
        return 0    
    def plotCDF(self):
        SelectedColumn = self.getSelectedColumns()
        if len(SelectedColumn)!=1: return QtWidgets.QMessageBox.critical(self, "错误", "请选择一列!")
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        SelectedDF = SelectedDF.iloc[:,0]
        tempFigDlg = _MatplotlibWidget()
        Fig = tempFigDlg.Mpl.Fig
        Axes = Fig.add_subplot(111)
        xData = SelectedDF[pd.notnull(SelectedDF)].values
        xData.sort()
        nData = len(xData)
        Delta = (xData[-1]-xData[0])/nData
        xData = np.append(xData[0]-Delta,xData)
        xData = np.append(xData,xData[-1]+Delta)
        yData = (np.linspace(0,nData+1,nData+2))/(nData)
        yData[-1] = yData[-2]
        Axes.plot(xData,yData,label='经验分布函数',linewidth=2,color='b')
        xNormalData = np.linspace(xData[0],xData[-1],(nData+2)*10)
        yNormalData = stats.norm.cdf(xNormalData,loc=np.mean(xData[1:-1]),scale=np.std(xData[1:-1]))
        Axes.plot(xNormalData, yNormalData, label='Normal Distribution', linewidth=2, color='r')
        Axes.legend(loc='upper left',shadow=True)
        tempFigDlg.Mpl.draw()
        tempFigDlg.show()
        return 0
    def plotScatter(self):
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        elif (SelectedDF.shape[1]<1) or (SelectedDF.shape[1]>3): return QtWidgets.QMessageBox.critical(self, "错误", "请选择一到三列!")
        isOK = QtWidgets.QMessageBox.question(self, "回归线", "是否添加回归线?", QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Cancel)
        tempFigDlg = _MatplotlibWidget()
        Fig = tempFigDlg.Mpl.Fig
        Axes = Fig.add_subplot(111)
        SelectedDF.dropna()
        if SelectedDF.shape[1]==1:
            xData = np.linspace(0, SelectedDF.shape[0]-1, SelectedDF.shape[0])
            yData = SelectedDF.iloc[:,0].values
            Axes.scatter(xData, yData, label="散点图", color='b')
        if SelectedDF.shape[1]==2:
            xData = SelectedDF.iloc[:,0].values
            yData = SelectedDF.iloc[:,1].values
            Axes.scatter(xData, yData, label="散点图", color='b')
        elif SelectedDF.shape[1]==3:
            xData = SelectedDF.iloc[:,0].values
            yData = SelectedDF.iloc[:,1].values
            zData = SelectedDF.iloc[:,2].astype('float')
            Size = ((zData-zData.mean()) / zData.std()*50).values
            Color = ((zData-zData.min()) / (zData.max()-zData.min())).values
            Axes.scatter(xData, yData, s=Size, c=Color, label="散点图")
        if isOK==QtWidgets.QMessageBox.Ok:
            Result = sm.OLS(yData, sm.add_constant(xData, prepend=True), missing='drop').fit()
            xData.sort()
            yRegressData = Result.params[0] + Result.params[1] * xData
            Axes.plot(xData, yRegressData, label="回归线", color='r', linewidth=2)
        Axes.legend(loc="upper left", shadow=True)
        tempFigDlg.Mpl.draw()
        tempFigDlg.show()
        return 0
    def plotScatter3D(self):
        from mpl_toolkits.mplot3d import Axes3D
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        elif SelectedDF.shape[1]!=3: return QtWidgets.QMessageBox.critical(self, '错误', '请选择三列!')
        isOK = QtWidgets.QMessageBox.question(self, "添加回归面", "是否添加回归面?", QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Cancel)
        tempFigDlg = _MatplotlibWidget()
        Fig = tempFigDlg.Mpl.Fig
        Axes = Axes3D(Fig)
        SelectedDF.dropna()
        xData = SelectedDF.iloc[:,0].values
        yData = SelectedDF.iloc[:,1].values
        zData = SelectedDF.iloc[:,2].values
        Axes.scatter(xData, yData, zData, label='散点图')
        if isOK==QtWidgets.QMessageBox.Ok:
            Result = sm.OLS(zData, sm.add_constant(np.c_[np.ones(SelectedDF.shape[0]), xData, yData], prepend=True), missing='drop').fit()
            X, Y = np.meshgrid(np.sort(xData), np.sort(yData))
            zRegressData = Result.params[0] + Result.params[1]*X + Result.params[2]*Y
            Axes.plot_surface(X, Y, zRegressData, label='回归面')
        tempFigDlg.Mpl.draw()
        tempFigDlg.show()
        return 0
    def plotCorr(self):# 相关图
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        elif (SelectedDF.shape[1]<2): return QtWidgets.QMessageBox.critical(self, "错误", "请选择至少两列!")
        PairGrid = sns.pairplot(SelectedDF, kind='reg', dropna=True)
        tempFigDlg = _MatplotlibWidget(fig=PairGrid.fig)
        tempFigDlg.Mpl.draw()
        tempFigDlg.show()
        return 0
    def plotJoint(self):# 联合图
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        elif (SelectedDF.shape[1]!=2): return QtWidgets.QMessageBox.critical(self, "错误", "请选择两列!")
        JointGrid = sns.jointplot(x=SelectedDF.columns[0], y=SelectedDF.columns[1], data=SelectedDF, kind='reg', dropna=True)
        tempFigDlg = _MatplotlibWidget(fig=JointGrid.fig)
        tempFigDlg.Mpl.draw()
        tempFigDlg.show()
        return 0
    def plotQQ(self):# QQ 图
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        elif (SelectedDF.shape[1]<1) or (SelectedDF.shape[1]>2): return QtWidgets.QMessageBox.critical(self, "错误", "请选择一列或者两列!")
        RefLine, isOK = QtWidgets.QInputDialog.getItem(self, "参考线", "参考线", ["q", "45", "s", "r", "无"])
        if not isOK: return 0
        if RefLine=="无": RefLine = None
        tempFigDlg = _MatplotlibWidget()
        Fig = tempFigDlg.Mpl.Fig
        Axes = Fig.add_subplot(111)
        SelectedDF.dropna()
        if SelectedDF.shape[1]==1:
            DistNames, _ = stats._continuous_distns.get_distribution_names(list(vars(stats).items()), stats.rv_continuous)
            DistNames.remove("norm")
            DistNames = ["norm"]+sorted(DistNames)
            Dist, isOK = QtWidgets.QInputDialog.getItem(self, "理论分布", "理论分布", DistNames)
            if not isOK: return 0
            sm.qqplot(data=SelectedDF.iloc[:,0].values, dist=eval("stats."+Dist), fit=True, line=RefLine, ax=Axes)
        else:
            pp_x = sm.ProbPlot(SelectedDF.iloc[:,0].values, fit=True)
            pp_y = sm.ProbPlot(SelectedDF.iloc[:,1].values, fit=True)
            pp_x.qqplot(xlabel="Sample Quantiles of "+str(SelectedDF.columns[0]), ylabel="Sample Quantiles of "+str(SelectedDF.columns[1]), other=pp_y, line=RefLine, ax=Axes)
        tempFigDlg.Mpl.draw()
        tempFigDlg.show()
        return 0
    def plotRadar(self):# 雷达图
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        Data = SelectedDF.values
        Min, isOK = QtWidgets.QInputDialog.getDouble(self, "最小值", "最小值: ", np.nanmin(Data))
        if not isOK: return 0
        Max, isOK = QtWidgets.QInputDialog.getDouble(self, "最大值", "最大值: ", np.nanmax(Data))
        if not isOK: return 0
        Angles = np.linspace(0, 2*np.pi, SelectedDF.shape[0], endpoint=False)
        Angles = np.concatenate((Angles, [Angles[0]]))
        Data = np.concatenate((Data, [Data[0, :]]))
        tempFigDlg = _MatplotlibWidget()
        Fig = tempFigDlg.Mpl.Fig
        Axes = Fig.add_subplot(111, polar=True)
        Axes.set_thetagrids(Angles*180/np.pi, SelectedDF.index.values)#设置网格标签
        Axes.set_rlim(Min, Max)# 设置显示的极径范围
        for i in range(Data.shape[1]): Axes.plot(Angles, Data[:, i], "o-")
        Axes.set_theta_zero_location("NW")#设置极坐标0°位置
        Axes.set_rlabel_position(255)#设置极径标签位置
        tempFigDlg.Mpl.draw()
        tempFigDlg.show()
        return 0
    def plotHeatMap(self):
        SelectedDF, Msg = self.getSelectedDF(all_num=True)
        if SelectedDF is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        SelectedIndex = self._getDataIndex(list(SelectedDF.index))
        SelectedDF = SelectedDF.loc[SelectedIndex]
        SelectedIndex = [str(iIndex) for iIndex in SelectedIndex]
        tempFigDlg = _MatplotlibWidget()
        Fig = tempFigDlg.Mpl.Fig
        Axes = Fig.add_subplot(111)
        HeatMap = Axes.pcolor(SelectedDF.astype('float').values, cmap=matplotlib.cm.Reds)
        Axes.set_xticks(np.arange(SelectedDF.shape[0])+0.5, minor=False)
        Axes.set_yticks(np.arange(SelectedDF.shape[1])+0.5, minor=False)
        Axes.invert_yaxis()
        Axes.xaxis.tick_top()
        Axes.set_xticklabels(SelectedIndex, minor=False)
        Axes.set_yticklabels(list(SelectedDF.columns), minor=False)
        tempFigDlg.Mpl.draw()
        tempFigDlg.show()
        return 0
    def _getPlotArgs(self, plot_data):
        nCol = plot_data.shape[1]
        PlotMode = ['Line']*nCol
        PlotAxes = ['左轴']*nCol
        PlotArgs = [{} for i in range(nCol)]
        DataTable = QtWidgets.QTableWidget()
        DataTable.setRowCount(nCol)
        DataTable.setVerticalHeaderLabels([str(iCol) for iCol in plot_data.columns])
        DataTable.setColumnCount(6)
        DataTable.setHorizontalHeaderLabels(['图像模式','坐标轴','颜色','线性','标记','线宽'])
        AllModes = ['Line','Bar','Stack']
        AllAxes = ['左轴','右轴']
        AllLineStyles = ['-','--','-.',':']
        AllColor = ['默认','blue','red','cyan','green','black','magenta','yellow','white']
        AllMarkers = ['默认','.',',','+','x','*','o','v','^','<','>','1','2','3','4','s','p','h','H','D','d','|','-']
        for i in range(nCol):
            iComboBox = QtWidgets.QComboBox(None)
            iComboBox.addItems(AllModes)
            DataTable.setCellWidget(i,0,iComboBox)
            iComboBox = QtWidgets.QComboBox(None)
            iComboBox.addItems(AllAxes)
            DataTable.setCellWidget(i,1,iComboBox)
            iComboBox = QtWidgets.QComboBox(None)
            iComboBox.addItems(AllColor)
            DataTable.setCellWidget(i,2,iComboBox)
            iComboBox = QtWidgets.QComboBox(None)
            iComboBox.addItems(AllLineStyles)
            DataTable.setCellWidget(i,3,iComboBox)
            iComboBox = QtWidgets.QComboBox(None)
            iComboBox.addItems(AllMarkers)
            DataTable.setCellWidget(i,4,iComboBox)
            iComboBox = QtWidgets.QDoubleSpinBox(None)
            iComboBox.setRange(0.1,10)
            iComboBox.setSingleStep(0.1)
            iComboBox.setValue(2.0)
            DataTable.setCellWidget(i,5,iComboBox)
        Dlg = _TableDlg(None, DataTable)
        Dlg.exec_()
        if not Dlg.isOK: return (None, None, None)
        for i in range(nCol):
            PlotMode[i] = DataTable.cellWidget(i,0).currentText()
            PlotAxes[i] = DataTable.cellWidget(i,1).currentText()
            iColor = DataTable.cellWidget(i,2).currentText()
            if iColor=="默认":
                if PlotMode[i] in ("Bar", "Stack"): PlotArgs[i]['color'] = "b"
            else:
                PlotArgs[i]['color'] = iColor
            PlotArgs[i]['linestyle'] = DataTable.cellWidget(i,3).currentText()
            iMarker = DataTable.cellWidget(i,4).currentText()
            if iMarker!='默认':
                PlotArgs[i]['marker'] = iMarker
            PlotArgs[i]['linewidth'] = DataTable.cellWidget(i,5).value()
        return (PlotMode, PlotAxes, PlotArgs)
    @QtCore.pyqtSlot()
    def on_PlotButton_clicked(self):
        # 获取绘图数据
        PlotResult, Msg = self.getSelectedDF(all_num=True)
        if PlotResult is None: return QtWidgets.QMessageBox.critical(self, "错误", Msg)
        # 设置绘图模式
        PlotMode, PlotAxes, PlotArgs = self._getPlotArgs(PlotResult)
        if PlotMode is None: return 0
        # 设置要绘制的索引
        xData = self._getDataIndex(PlotResult.index)
        if not xData: return QtWidgets.QMessageBox.critical(self, "错误", "绘图数据为空!")
        PlotResult = PlotResult.loc[xData]
        if (not PlotResult.index.is_mixed()) and isinstance(PlotResult.index[0], (dt.datetime, dt.date)):# index 是日期或者时间
            isDT = True
            xData = np.arange(0, PlotResult.shape[0])
            xTicks = np.arange(0, PlotResult.shape[0], max(1, int(PlotResult.shape[0]/10)))
            if isinstance(PlotResult.index[0], dt.datetime):
                if float(np.min(np.diff(PlotResult.index)) / 1e9 / 3600 / 24)>=1:
                    xTickLabels = [PlotResult.index[i].strftime("%Y-%m-%d") for i in xTicks]
                else:
                    xTickLabels = [PlotResult.index[i].strftime("%Y-%m-%d %H:%M:%S.%f") for i in xTicks]
            else:
                xTickLabels = [PlotResult.index[i].strftime("%Y-%m-%d") for i in xTicks]
        else:
            isDT = False
            xTickLabels = []
            isStr = False
            for iData in xData:
                xTickLabels.append(str(iData))
                if isinstance(iData, str): isStr = True
            if isStr: xData = np.arange(0, PlotResult.shape[0])
        FigDlg = _MatplotlibWidget()
        Fig = FigDlg.Mpl.Fig
        nLeftAxe, nRightAxe = PlotAxes.count("左轴"), PlotAxes.count("右轴")
        if nLeftAxe>0:
            LeftAxe = Fig.add_subplot(111)
            if nRightAxe>0:
                RightAxe = LeftAxe.twinx()
                LeftAxe.set_prop_cycle(cycler('color', [plt.cm.Spectral(i) for i in np.arange(0, nLeftAxe)/PlotResult.shape[1]]))
                RightAxe.set_prop_cycle(cycler('color', [plt.cm.Spectral(i) for i in np.arange(nLeftAxe, PlotResult.shape[1])/PlotResult.shape[1]]))
            else:
                LeftAxe.set_prop_cycle(cycler('color', [plt.cm.Spectral(i) for i in np.linspace(0, 1, PlotResult.shape[1])]))
        else:
            RightAxe = Fig.add_subplot(111)
            LeftAxe = RightAxe
            LeftAxe.set_prop_cycle(cycler('color', [plt.cm.Spectral(i) for i in np.linspace(0, 1, PlotResult.shape[1])]))
        for i in range(PlotResult.shape[1]):
            iAxe = (LeftAxe if PlotAxes[i]=="左轴" else RightAxe)
            yData = PlotResult.iloc[:,i]
            if not isDT:
                try:
                    if PlotMode[i]=="Line":
                        iAxe.plot(yData, label=str(yData.name), **PlotArgs[i])
                    elif PlotMode[i]=="Bar":
                        iAxe.bar(yData.index, yData.values, label=str(yData.name), **PlotArgs[i])
                    elif PlotMode[i]=="Stack":
                        iAxe.stackplot(yData.index, yData.values, **PlotArgs[i])
                except:
                    if PlotMode[i]=="Line":
                        iAxe.plot(xData, yData.values, label=str(yData.name), **PlotArgs[i])
                    elif PlotMode[i]=="Bar":
                        iAxe.bar(xData, yData.values, label=str(yData.name), **PlotArgs[i])
                    elif PlotMode[i]=="Stack":
                        iAxe.stackplot(xData, yData.values, **PlotArgs[i])
                    iAxe.set_xticks(xData)
                    iAxe.set_xticklabels(xTickLabels)
            else:
                if PlotMode[i]=="Line":
                    iAxe.plot(xData, yData.values, label=str(yData.name), **PlotArgs[i])
                elif PlotMode[i]=="Bar":
                    iAxe.bar(xData, yData.values, label=str(yData.name), **PlotArgs[i])
                elif PlotMode[i]=="Stack":
                    iAxe.stackplot(xData, yData.values, **PlotArgs[i])
                iAxe.set_xticks(xTicks)
                iAxe.set_xticklabels(xTickLabels)
        if nLeftAxe>0: LeftAxe.legend(loc='upper left',shadow=True)
        if nRightAxe>0: RightAxe.legend(loc='upper right',shadow=True)
        plt.title(','.join([str(iCol) for iCol in PlotResult.columns]))
        FigDlg.Mpl.draw()
        FigDlg.show()
        return 0

if __name__=='__main__':
    # 测试代码
    from QuantStudio.Tools.DateTimeFun import getDateSeries
    
    Bar2 = pd.DataFrame(np.random.randn(3,2), index=["中文", "b2", "b3"], columns=["中文", "我是个例子"])
    Bar2.iloc[0,0] = np.nan
    Dates = getDateSeries(dt.date(2016,1,1), dt.date(2016,12,31))
    TestData = {"Bar1":{"a":{"a1":pd.DataFrame(np.random.rand(11,10),index=Dates[:11],columns=['a'+str(i) for i in range(10)]),
                             "a2":pd.DataFrame(np.random.rand(10,2))},
                        "b":pd.DataFrame(['a']*150,columns=['c'])},
                "Bar2": Bar2}
    app = QtWidgets.QApplication(sys.argv)
    #TestWindow = PlotlyResultDlg(None, TestData)
    TestWindow = MatplotlibResultDlg(None, TestData)
    TestWindow.show()
    app.exec_()
    sys.exit()