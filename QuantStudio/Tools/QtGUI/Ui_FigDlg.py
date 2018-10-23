# -*- coding: utf-8 -*-

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class _MplCanvas(FigureCanvas):
    def __init__(self):
        self.Fig = Figure()
        #self.Axe = self.Fig.add_subplot(111)
        FigureCanvas.__init__(self, self.Fig)
        FigureCanvas.setSizePolicy(self,QtGui.QSizePolicy.Expanding,QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class _PlotWidget(QtGui.QWidget):
    def __init__(self,parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.Canvas = _MplCanvas()
        self.vbl = QtGui.QVBoxLayout()
        self.vbl.addWidget(self.Canvas)
        self.setLayout(self.vbl)

class Ui_FigDlg(QtGui.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
    def setupUi(self, FigDlg):
        FigDlg.setObjectName(_fromUtf8("FigDlg"))
        FigDlg.resize(934, 611)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(FigDlg.sizePolicy().hasHeightForWidth())
        FigDlg.setSizePolicy(sizePolicy)
        FigDlg.setSizeGripEnabled(True)
        self.horizontalLayout = QtGui.QHBoxLayout(FigDlg)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.PlotWidget = _PlotWidget(FigDlg)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PlotWidget.sizePolicy().hasHeightForWidth())
        self.PlotWidget.setSizePolicy(sizePolicy)
        self.PlotWidget.setObjectName(_fromUtf8("PlotWidget"))
        self.horizontalLayout.addWidget(self.PlotWidget)

        self.retranslateUi(FigDlg)
        QtCore.QMetaObject.connectSlotsByName(FigDlg)

    def retranslateUi(self, FigDlg):
        FigDlg.setWindowTitle(_translate("FigDlg", "图像", None))

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    FigDlg = QtGui.QDialog()
    ui = Ui_FigDlg()
    ui.setupUi(FigDlg)
    FigDlg.show()
    sys.exit(app.exec_())

