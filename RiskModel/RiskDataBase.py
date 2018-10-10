# coding=utf-8
import os
import datetime as dt

import numpy as np
import pandas as pd
from traits.api import Str

from QuantStudio.Tools.AuxiliaryFun import genAvailableName
from QuantStudio.Tools.FileFun import listDirDir, readJSONFile
from QuantStudio.Tools.DateTimeFun import cutDateTime
from .RiskModelFun import dropRiskMatrixNA, decomposeCov2Corr
from QuantStudio import __QS_Object__

# 风险数据库基类, 必须存储的数据有:
# 风险矩阵: Cov, Panel(items=[时点], major_axis=[ID], minor_axis=[ID])
class RiskDataBase(__QS_Object__):
    """风险数据库"""
    Name = Str("风险数据库")
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._DBType = "RDB"
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    # 链接数据库
    def connect(self):
        return 0
    # 断开风险数据库
    def disconnect(self):
        return 0
    # 检查数据库是否可用
    def isAvailable(self):
        return False
    # -------------------------------表的管理---------------------------------
    # 获取数据库中的表名
    @property
    def TableNames(self):
        return []
    # 获取表的元数据
    def getTableMetaData(self, table_name, key=None):
        if key is None: return pd.Series()
        return None
    # 设置表的元数据
    def setTableMetaData(self, table_name, key=None, value=None, meta_data=None):
        return 0
    # 重命名表
    def renameTable(self, old_table_name, new_table_name):
        return 0
    # 删除表
    def deleteTable(self, table_name):
        return 0
    # 获取这张表中的时点序列
    def getTableDateTime(self, table_name, start_dt=None, end_dt=None):
        return []
    # 获取这张表中的 ID 序列
    def getTableID(self, table_name, idt=None):
        if idt is None: idt = self.getTableDateTime(table_name)[-1]
        Cov = self.readCov(table_name, dts=[idt])
        return Cov.major_axis.tolist()
    # 删除一张表中的某些时点
    def deleteDateTime(self, table_name, dts):
        return 0
    # ------------------------数据读取--------------------------------------
    # 读取协方差矩阵, Panel(items=[时点], major_axis=[ID], minor_axis=[ID])
    def readCov(self, table_name, dts, ids=None):
        return None
    # 读取相关系数矩阵, Panel(items=[时点], major_axis=[ID], minor_axis=[ID])
    def readCorr(self, table_name, dts, ids=None):
        Cov = self.readCov(table_name, dts=dts, ids=ids)
        Corr = {}
        for iDT in Cov.items:
            iCov = Cov.loc[iDT]
            iCorr, _ = decomposeCov2Corr(iCov.values)
            Corr[iDT] = pd.DataFrame(iCorr, index=iCov.index, columns=iCov.columns)
        return pd.Panel(Corr).loc[Cov.items]
    # ------------------------数据存储--------------------------------------
    # 存储数据
    def writeData(self, table_name, idt, icov):
        return 0
# 多因子风险数据库基类, 即风险矩阵可以分解成 V=X*F*X'+D 的模型, 其中 D 是对角矩阵, 必须存储的数据有:
# 因子风险矩阵: FactorCov(F), Panel(items=[时点], major_axis=[因子], minor_axis=[因子])
# 特异性风险: SpecificRisk(D), DataFrame(index=[时点], columns=[ID])
# 因子截面数据: FactorData(X), Panel(items=[因子], major_axis=[时点], minor_axis=[ID])
# 因子收益率: FactorReturn, DataFrame(index=[时点], columns=[因子])
# 特异性收益率: SpecificReturn, DataFrame(index=[时点], columns=[ID])
# 可选存储的数据有:
# 回归统计量: Statistics, {"tValue":Series(data=统计量, index=[因子]),"FValue":double,"rSquared":double,"rSquared_Adj":double}
class FactorRDB(RiskDataBase):
    """多因子风险数据库"""
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
        self._DBType = "FRDB"
        return
    # 获取表的所有因子
    def getTableFactor(self, table_name):
        return []
    def getTableID(self, table_name, idt=None):
        if idt is None: idt = self.getTableDateTime(table_name)[-1]
        SpecificRisk = self.readSpecificRisk(table_name, dts=[idt])
        return SpecificRisk.index.tolist()
    # 获取因子收益的时点
    def getFactorReturnDateTime(self, table_name, start_dt=None, end_dt=None):
        FactorReturn = self.readFactorReturn(table_name)
        if FactorReturn is not None: return cutDateTime(FactorReturn.index, start_dt=start_dt, end_dt=end_dt)
        return []
    # 获取特异性收益的时点
    def getSpecificReturnDateTime(self, table_name, start_dt=None, end_dt=None):
        SpecificReturn = self.readSpecificReturn(table_name)
        if SpecificReturn is not None: return cutDateTime(SpecificReturn.index, start_dt=start_dt, end_dt=end_dt)
        return []
    def readCov(self, table_name, dts, ids=None):
        FactorCov = self.readFactorCov(table_name, dts=dts)
        FactorData = self.readFactorData(table_name, dts=dts, ids=ids)
        SpecificRisk = self.readSpecificRisk(table_name, dts=dts, ids=ids)
        Data = {}
        for iDT in FactorCov:
            if ids is None:
                iIDs = SpecificRisk.loc[iDT].index
                iFactorData = FactorData[iDT].loc[iIDs]
            else:
                iIDs = ids
                iFactorData = FactorData[iDT]
            iCov = np.dot(np.dot(iFactorData.values, FactorCov[iDT].values), iFactorData.values.T) + np.diag(SpecificRisk.loc[iDT].values**2)
            Data[iDT] = pd.DataFrame(iCov, index=iIDs, columns=iIDs)
        return pd.Panel(Data).loc[dts]
    # 读取因子风险矩阵
    def readFactorCov(self, table_name, dts):
        return None
    # 读取特异性风险
    def readSpecificRisk(self, table_name, dts, ids=None):
        return None
    # 读取截面数据
    def readFactorData(self, table_name, dts, ids=None):
        return None
    # 读取因子收益率
    def readFactorReturn(self, table_name, dts):
        return None
    # 读取残余收益率
    def readSpecificReturn(self, table_name, dts, ids=None):
        return None
    # 存储数据
    def writeData(self, table_name, idt, factor_data=None, factor_cov=None, specific_risk=None, factor_ret=None, specific_ret=None, **kwargs):
        return 0