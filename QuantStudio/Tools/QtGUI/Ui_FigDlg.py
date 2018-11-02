# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class _MplCanvas(FigureCanvas):
    def __init__(self):
        self.Fig = Figure()
        #self.Axe = self.Fig.add_subplot(111)
        FigureCanvas.__init__(self, self.Fig)
        FigureCanvas.setSizePolicy(self,QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class _PlotWidget(QtWidgets.QWidget):
    def __init__(self,parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.Canvas = _MplCanvas()
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.Canvas)
        self.setLayout(self.vbl)

class Ui_FigDlg(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
    def setupUi(self, FigDlg):
        FigDlg.setObjectName("FigDlg")
        FigDlg.resize(934, 611)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(FigDlg.sizePolicy().hasHeightForWidth())
        FigDlg.setSizePolicy(sizePolicy)
        FigDlg.setSizeGripEnabled(True)
        self.horizontalLayout = QtWidgets.QHBoxLayout(FigDlg)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.PlotWidget = _PlotWidget(FigDlg)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PlotWidget.sizePolicy().hasHeightForWidth())
        self.PlotWidget.setSizePolicy(sizePolicy)
        self.PlotWidget.setObjectName("PlotWidget")
        self.horizontalLayout.addWidget(self.PlotWidget)

        self.retranslateUi(FigDlg)
        QtCore.QMetaObject.connectSlotsByName(FigDlg)

    def retranslateUi(self, FigDlg):
        _translate = QtCore.QCoreApplication.translate
        FigDlg.setWindowTitle(_translate("FigDlg", "图像", None))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    FigDlg = QtWidgets.QDialog()
    ui = Ui_FigDlg()
    ui.setupUi(FigDlg)
    FigDlg.show()
    sys.exit(app.exec_())

