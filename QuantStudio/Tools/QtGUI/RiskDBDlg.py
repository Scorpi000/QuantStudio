# -*- coding: utf-8 -*-
import shutil
import os

import numpy as np
import pandas as pd
from progressbar import ProgressBar
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QDialog

from QuantStudio.Tools.QtGUI.Ui_RiskDBDlg import Ui_RiskDBDlg
from QuantStudio.Tools.QtGUI.PreviewFactorDlg import PreviewDlg
from QuantStudio.Tools.FileFun import listDirFile, loadCSVFactorData
from QuantStudio.Tools.AuxiliaryFun import genAvailableName
from QuantStudio.Tools.DataTypeConversionFun import DataFrame2Series

class RiskDBDlg(QDialog, Ui_RiskDBDlg):
    def __init__(self, risk_db, parent=None):
        super(RiskDBDlg, self).__init__(parent)
        self.setupUi(self)
        self.RiskDB = risk_db
        self.populateRiskDBTree()
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
    @pyqtSlot()
    def on_UpdateButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError
    
    @pyqtSlot()
    def on_DescriptionButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError
    
    @pyqtSlot()
    def on_ViewButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError
    
    @pyqtSlot()
    def on_RenameButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError
    
    @pyqtSlot()
    def on_DeleteButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError
    
    @pyqtSlot()
    def on_MoveButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError
    
    @pyqtSlot()
    def on_CSVExportButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError
    
    @pyqtSlot()
    def on_CSVImportButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError
