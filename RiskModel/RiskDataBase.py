# coding=utf-8
import os
import shelve
import shutil
import datetime as dt
import zipfile
from multiprocessing import Lock

import numpy as np
import pandas as pd
from traits.api import Str, Directory

from QuantStudio.Tools.AuxiliaryFun import genAvailableName
from QuantStudio.Tools.FileFun import listDirDir, readJSONFile
from QuantStudio.Tools.DateTimeFun import cutDateTime
from .RiskModelFun import dropRiskMatrixNA, decomposeCov2Corr
from QuantStudio import __QS_Object__, __QS_Error__, __QS_LibPath__, __QS_CacheLock__, __QS_CachePath__

# 风险数据库基类, 必须存储的数据有:
# 风险矩阵: Cov, DataFrame(data=协方差, index=[ID], columns=[ID])
class RiskDataBase(__QS_Object__):
    """RiskDataBase"""
    Name = Str("风险数据库")
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self.DBType = "RDB"
        super().__init__(sys_args=sys_args, config_file=config_file)
        self._TableDT = {}#{表名：[时点]}
        self._isAvailable = False
        return
    # 链接数据库
    def connect(self):
        self._isAvailable = True
        return 0
    # 断开风险数据库
    def disconnect(self):
        self._isAvailable = False
        return 0
    # 检查数据库是否可用
    def isAvailable(self):
        return self._isAvailable
    # -------------------------------表的管理---------------------------------
    # 获取数据库中的表名
    @property
    def TableNames(self):
        return sorted(self._TableDT)
    # 获取一个可用的表名
    def genAvailableTableName(self, header='NewRiskData'):
        return genAvailableName(header, self._TableDT)
    # 检查表是否存在
    def checkTableExistence(self, table_name):
        return (table_name in self._TableDT)
    # 获取表的元数据
    def getTableMetaData(self, table_name, key=None):
        if key is None: return {}
        return None
    # 设置表的元数据
    def setTableMetaData(self, table_name, key=None, value=None, meta_data=None):
        return 0
    # 创建表, dts:[时点], 前提是当前不存在表 table_name, 否则行为不可预知. 
    def createTable(self, table_name, dts):
        self._TableDT[table_name] = dts
        return 0
    # 重命名表
    def renameTable(self, old_table_name, new_table_name):
        self._TableDT[new_table_name] = self._TableDT.pop(old_table_name)
        return 0
    # 删除表
    def deleteTable(self, table_name):
        self._TableDT.pop(table_name, None)
        return 0
    # 获取这张表中的时点序列
    def getTableDateTime(self, table_name, start_dt=None, end_dt=None):
        return cutDateTime(self._TableDT[table_name], start_dt, end_dt)
    # 获取这张表中的 ID 序列
    def getTableID(self, table_name, idt=None):
        if idt is None: idt = self._TableDT[table_name][-1]
        Cov = self.readCov(table_name, dts=idt)
        return Cov.index.tolist()
    # 复制表
    def copyTable(self, table_name, new_table_name):
        self._TableDT[new_table_name] = self._TableDT[table_name]
        return 0
    # 删除一张表中的某些时点
    def deleteDateTime(self, table_name, dts):
        for iDT in dts:
            if iDT in self._TableDT[table_name]: self._TableDT[table_name].remove(iDT)
        return 0
    # ------------------------数据读取--------------------------------------
    # 加载风险矩阵数据到内存中
    def readCov(self, table_name, dts=None, ids=None, drop_na=False):
        return None
    # 加载相关系数矩阵数据到内存中
    def readCorr(self, table_name, dts=None, ids=None, drop_na=False):
        Cov = self.readCov(table_name, dts=dts, ids=ids, drop_na=drop_na)
        if Cov is None: return None
        if not isinstance(Cov, dict):
            Corr, _ = decomposeCov2Corr(Cov.values)
            return pd.DataFrame(Corr, index=Cov.index, columns=Cov.columns)
        Corr = {}
        for iDT, iCov in Cov.items():
            iCorr, _ = decomposeCov2Corr(iCov.values)
            Corr[iDT] = pd.DataFrame(iCorr, index=iCov.index, columns=iCov.columns)
        return Corr
    # ------------------------数据存储--------------------------------------
    # 存储数据
    def writeData(self, table_name, idt, cov=None):
        return 0
    # 生成数据补丁
    def genDSDataPatch(self, file_path, tables, start_dt=None, end_dt=None):
        return 0
    # 用压缩补丁文件更新数据
    def updateDSData(self, zip_file_path):
        return 0
# 建立于 shelve 模块上的风险数据库, 以硬盘中的 shelve 文件作为数据存储方式.
# 每个 shelve 文件以时点为索引, 每个时点对应相应的数据
class ShelveRDB(RiskDataBase):
    """RDB"""
    MainDir = Directory(label="主目录", arg_type="Directory", order=0)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
        self._DataLock = Lock()
        self._Suffix = ("nt" if os.name=="nt" else "")
    def __QS_initArgs__(self):
        Config = readJSONFile(__QS_LibPath__+os.sep+"RDBConfig.json")
        for iArgName, iArgVal in Config.items(): self[iArgName] = iArgVal
    def connect(self):
        if not os.path.isdir(self.MainDir): raise __QS_Error__("不存在 ShelveRDB 的主目录: %s!" % self.MainDir)
        AllTables = listDirDir(self.MainDir)
        self._TableDT = {}#{表名：[时点]}
        with self._DataLock:
            for iTable in AllTables:
                iTablePath = self.MainDir+os.sep+iTable
                with shelve.open(iTablePath+os.sep+"__TableInfo") as iTableInfoFile:
                    iType = iTableInfoFile.get("DBType")
                    if iType is None:
                        if os.path.isfile(iTablePath+os.sep+"Cov"+("."+self._Suffix if self._Suffix!="" else "")):
                            iType = "RDB"
                        else:
                            continue
                        iTableInfoFile["DBType"] = iType
                if iType=="RDB":
                    with shelve.open(iTablePath+os.sep+"Cov") as iFile:
                        iDTs = sorted(iFile)
                    self._TableDT[iTable] = [dt.datetime.strptime(ijDT, "%Y-%m-%d %H:%M:%S.%f") for ijDT in iDTs]
        self._isAvailable = True
        return 0
    # -------------------------------表的管理---------------------------------
    def getTableMetaData(self, table_name, key=None):
        with self._DataLock:
            with shelve.open(self.MainDir+os.sep+table_name+os.sep+"__TableInfo") as TableInfoFile:
                if key is not None: return TableInfoFile.get(key, None)
                return {iKey:TableInfoFile[iKey] for iKey in TableInfoFile}
    def setTableMetaData(self, table_name, key=None, value=None, meta_data=None):
        with self._DataLock:
            with shelve.open(self.MainDir+os.sep+table_name+os.sep+"__TableInfo") as TableInfoFile:
                if key is not None: TableInfoFile[key] = value
                if meta_data is not None:
                    for iKey, iVal in meta_data.items():
                        TableInfoFile[iKey] = iVal
        return 0
    def createTable(self, table_name, dts):
        self._TableDT[table_name] = dts
        with self._DataLock:
            os.mkdir(self.MainDir+os.sep+table_name)
        return 0
    def renameTable(self, old_table_name, new_table_name):
        self._TableDT[new_table_name] = self._TableDT.pop(old_table_name)
        with self._DataLock:
            os.rename(self.MainDir+os.sep+old_table_name, self.MainDir+os.sep+new_table_name)
        return 0
    def deleteTable(self, table_name):
        self._TableDT.pop(table_name, None)
        if not os.path.isdir(self.MainDir+os.sep+table_name): return 0
        with self._DataLock:
            shutil.rmtree(self.MainDir+os.sep+table_name, ignore_errors=True)
        return 0
    def copyTable(self, table_name, new_table_name):
        FactorRDB.copyTable(self, table_name, new_table_name)
        SrcDirPath = self.MainDir+os.sep+table_name
        DstDirPath = self.MainDir+os.sep+new_table_name
        shutil.copytree(SrcDirPath, DstDirPath)
        return 0
    def deleteDateTime(self, table_name, dts):
        with self._DataLock:
            with shelve.open(self.MainDir+os.sep+table_name+os.sep+"Cov") as CovFile:
                for iDT in dts:
                    if iDT in self._TableDT[table_name]: self._TableDT[table_name].remove(iDT)
                    CovFile.pop(iDT.strftime("%Y-%m-%d %H:%M:%S.%f"), None)
        return 0
    def readCov(self, table_name, dts=None, ids=None, drop_na=False):
        with self._DataLock:
            with shelve.open(self.MainDir+os.sep+table_name+os.sep+"Cov") as DataFile:
                if (dts is not None) and (not isinstance(dts,list)):
                    Data = DataFile.get(dts.strftime("%Y-%m-%d %H:%M:%S.%f"), None)
                    if Data is not None:
                        if ids is not None:
                            if Data.index.intersection(ids).shape[0]>0:
                                Data = Data.loc[ids, ids]
                            else:
                                Data = pd.DataFrame(index=ids, columns=ids)
                        if drop_na: Data = dropRiskMatrixNA(Data)
                else:
                    if dts is None:
                        dts = self._TableDT[table_name]
                    else:
                        dts = set(dts).intersection(set(self._TableDT[table_name]))
                    if not drop_na:
                        dropRiskMatrixNA = lambda x:x
                    if ids is None:
                        Data = {iDT:dropRiskMatrixNA(DataFile[iDT.strftime("%Y-%m-%d %H:%M:%S.%f")]) for iDT in dts}
                    else:
                        Data = {iDate:dropRiskMatrixNA(DataFile[iDT.strftime("%Y-%m-%d %H:%M:%S.%f")].loc[ids,ids]) for iDT in dts}
                    if Data=={}:
                        Data = None
        return Data
    def writeData(self, table_name, idt, cov=None, file_value={}):
        TablePath = self.MainDir+os.sep+table_name
        with self._DataLock:
            if table_name not in self._TableDT:
                self._TableDT[table_name] = []
                if not os.path.isdir(TablePath):
                    os.mkdir(TablePath)
            iDTStr = idt.strftime("%Y-%m-%d %H:%M:%S.%f")
            if cov is not None:
                with shelve.open(TablePath+os.sep+"Cov") as DataFile:
                    DataFile[iDTStr] = cov
                    self._TableDT[table_name].append(idt)
                self._TableDT[table_name].sort()
            for iFile, iValue in file_value.items():
                with shelve.open(TablePath+os.sep+iFile) as DataFile:
                    DataFile[iDTStr] = iValue
        return 0
    def genDSDataPatch(self, file_path, tables, start_dt=None, end_dt=None):
        with __QS_CacheLock__:
            PatchDir = __QS_CachePath__+os.sep+genAvailableName("RDataPatch", listDirDir(__QS_CachePath__))
            os.mkdir(PatchDir)
        PatchDB = ShelveRDB(sys_args={"主目录": PatchDir})
        PatchDB.connect()
        for iTableName in tables:
            iDTs = self.getTableDateTime(iTableName, start_dt=start_dt, end_dt=end_dt)
            iCov = self.readCov(iTableName, dts=iDTs)
            for jDT in iDTs:
                PatchDB.writeData(iTableName, jDT, cov=iCov.get(jDT, None))
        CWD = os.getcwd()
        os.chdir(PatchDir)
        ZipFile = zipfile.ZipFile(file_path, mode='a', compression=zipfile.ZIP_DEFLATED)
        for iTableName in tables:
            if os.path.isdir("."+os.sep+iTableName):
                iFiles = os.listdir("."+os.sep+iTableName)
            else:
                continue
            for jFile in iFiles:
                ZipFile.write("."+os.sep+iTableName+os.sep+jFile+("."+self._Suffix if self._Suffix!="" else ""))
        ZipFile.close()
        os.chdir(CWD)
        shutil.rmtree(PatchDir, ignore_errors=True)
        return 0
    def updateDSData(self, zip_file_path):
        ZipFile = zipfile.ZipFile(zip_file_path, 'r')
        with __QS_CacheLock__:
            PatchDir = __QS_CachePath__+os.sep+genAvailableName("RDataPatch", listDirDir(__QS_CachePath__))
            os.mkdir(PatchDir)
        for iFile in ZipFile.namelist():
            ZipFile.extract(iFile, PatchDir+os.sep)
        ZipFile.close()
        PatchDB = ShelveRDB(sys_args={"主目录":PatchDir})
        PatchDB.connect()
        for iTableName in PatchDB.TableNames:
            iDTs = PatchDB.getTableDateTime(iTableName)
            iCov = PatchDB.readCov(iTableName, dts=iDTs)
            for jDT in iDTs:
                self.writeData(iTableName, jDT, cov=iCov.get(jDT, None))
        shutil.rmtree(PatchDir, ignore_errors=True)
        return 0

# 多因子风险数据库基类, 即风险矩阵可以分解成 V=X*F*X'+D 的模型, 其中 D 是对角矩阵, 必须存储的数据有:
# 因子风险矩阵: FactorCov(F), DataFrame(data=协方差, index=[因子], columns=[因子])
# 特异性风险: SpecificRisk(D), Series(data=方差, index=[ID])
# 因子截面数据: FactorData(X), DataFrame(data=因子数据, index=[ID], columns=[因子])
# 因子收益率: FactorReturn, Series(data=收益率, index=[因子])
# 特异性收益率: SpecificReturn, Series(data=收益率, index=[ID])
# 可选存储的数据有:
# 回归统计量: Statistics, {"tValue":Series(data=统计量, index=[因子]),"FValue":double,"rSquared":double,"rSquared_Adj":double}
class FactorRDB(RiskDataBase):
    """FactorRDB"""
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
        self.DBType = "FRDB"
        return
    # 获取表的所有因子
    def getTableFactor(self, table_name):
        return []
    def getTableID(self, table_name, idt=None):
        if idt is None:
            idt = self._TableDT[table_name][-1]
        SpecificRisk = self.readSpecificRisk(table_name, dts=idt)
        return SpecificRisk.index.tolist()
    # 获取因子收益的时点
    def getFactorReturnDateTime(self, table_name, start_dt=None, end_dt=None):
        FactorReturn = self.readFactorReturn(table_name)
        if FactorReturn is not None:
            return cutDateTime(FactorReturn.index, start_dt=start_dt, end_dt=end_dt)
        else:
            return []
    # 获取特异性收益的时点
    def getSpecificReturnDateTime(self, table_name, start_dt=None, end_dt=None):
        SpecificReturn = self.readSpecificRisk(table_name)
        if SpecificReturn is not None:
            return cutDateTime(SpecificReturn.index, start_dt=start_dt, end_dt=end_dt)
        else:
            return []
    def readCov(self, table_name, dts=None, ids=None, drop_na=False):
        FactorCov = self.readFactorCov(table_name, dts=dts)
        FactorData = self.readFactorData(table_name, dts=dts, ids=ids)
        SpecificRisk = self.readSpecificRisk(table_name, dts=dts, ids=ids)
        if (dts is not None) and (not isinstance(dts,list)):
            if ids is None:
                ids = SpecificRisk.index
                FactorData = FactorData.loc[ids]
            CovMatrix = np.dot(np.dot(FactorData.values, FactorCov.values), FactorData.values.T) + np.diag(SpecificRisk.values**2)
            CovMatrix = pd.DataFrame(CovMatrix, index=ids, columns=ids)
            if drop_na:
                return dropRiskMatrixNA(CovMatrix)
            return CovMatrix
        Data = {}
        for iDT in FactorCov:
            if ids is None:
                iIDs = SpecificRisk.loc[iDT].index
                iFactorData = FactorData[iDT].loc[iIDs]
            else:
                iIDs = ids
                iFactorData = FactorData[iDT]
            iCovMatrix = np.dot(np.dot(iFactorData.values, FactorCov[iDT].values), iFactorData.values.T) + np.diag(SpecificRisk.loc[iDT].values**2)
            iCovMatrix = pd.DataFrame(iCovMatrix, index=iIDs, columns=iIDs)
            if drop_na:
                iCovMatrix = dropRiskMatrixNA(iCovMatrix)
            Data[iDT] = iCovMatrix
        return Data
    # 读取因子风险矩阵
    def readFactorCov(self, table_name, dts=None):
        return None
    # 读取特异性风险
    def readSpecificRisk(self, table_name, dts=None, ids=None):
        return None
    # 读取截面数据
    def readFactorData(self, table_name, dts=None, ids=None):
        return None
    # 读取因子收益率
    def readFactorReturn(self, table_name, dts=None):
        return None
    # 读取残余收益率
    def readSpecificReturn(self, table_name, dts=None, ids=None):
        return None
    # 存储数据
    def writeData(self, table_name, idt, factor_ret=None, specific_ret=None, factor_data=None, factor_cov=None, specific_risk=None):
        return 0

# 建立于 shelve 模块上的多因子风险数据库, 以硬盘中的 shelve 文件作为数据存储方式.
# 每个 shelve 文件以时点为索引, 每个时点对应相应的数据
class ShelveFRDB(ShelveRDB, FactorRDB):
    """FRDB"""
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
        self.DBType = 'FRDB'
        return
    def __QS_initArgs__(self):
        Config = readJSONFile(__QS_LibPath__+os.sep+"FRDBConfig.json")
        for iArgName, iArgVal in Config.items(): self[iArgName] = iArgVal
    def connect(self):
        if not os.path.isdir(self.MainDir): raise __QS_Error__("不存在 ShelveFRDB 的主目录: %s!" % self.MainDir)
        AllTables = listDirDir(self.MainDir)
        self._TableDT = {}#{表名：[时点]}
        with self._DataLock:
            for iTable in AllTables:
                iTablePath = self.MainDir+os.sep+iTable
                with shelve.open(iTablePath+os.sep+"__TableInfo") as iTableInfoFile:
                    iType = iTableInfoFile.get("DBType")
                    if iType is None:
                        if os.path.isfile(iTablePath+os.sep+"FactorCov"+("."+self._Suffix if self._Suffix!="" else "")):
                            iType = "FRDB"
                        else:
                            continue
                        iTableInfoFile["DBType"] = iType
                if iType=="FRDB":
                    with shelve.open(iTablePath+os.sep+"SpecificRisk") as iFile:
                        iDTs = sorted(iFile)
                    self._TableDT[iTable] = [dt.datetime.strptime(ijDT, "%Y-%m-%d %H:%M:%S.%f") for ijDT in iDTs]
        self._isAvailable = True
        return 0
    def getTableFactor(self, table_name):
        with self._DataLock:
            with shelve.open(self.MainDir+os.sep+table_name+os.sep+"FactorCov") as DataFile:
                FactorCov = DataFile[self._TableDT[table_name][-1]]
        return FactorCov.index.tolist()
    def getTableID(self, table_name, idt=None):
        return FactorRDB.getTableID(self, table_name, idt=idt)
    def getFactorReturnDateTime(self, table_name, start_dt=None, end_dt=None):
        with self._DataLock:
            with shelve.open(self.MainDir+os.sep+table_name+os.sep+"FactorReturn") as DataFile:
                DTs = sorted(DataFile)
        return cutDateTime(DTs, start_dt=start_dt, end_dt=end_dt)
    def getSpecificReturnDateTime(self, table_name, start_dt=None, end_dt=None):
        with self._DataLock:
            with shelve.open(self.MainDir+os.sep+table_name+os.sep+"SpecificReturn") as DataFile:
                DTs = sorted(DataFile)
        return cutDateTime(DTs, start_dt=start_dt, end_dt=end_dt)
    def deleteDateTime(self, table_name, dts):
        with self._DataLock:
            with shelve.open(self.MainDir+os.sep+table_name+os.sep+"FactorCov") as FactorCovFile:
                with shelve.open(self.MainDir+os.sep+table_name+os.sep+"SpecificRisk") as SpecificRiskFile:
                    for iDT in dts:
                        if iDT in self._TableDT[table_name]: self._TableDT[table_name].remove(iDT)
                        FactorCovFile.pop(iDT.strftime("%Y-%m-%d %H:%M:%S.%f"), None)
                        SpecificRiskFile.pop(iDT.strftime("%Y-%m-%d %H:%M:%S.%f"), None)
        return 0
    def readCov(self, table_name, dts=None, ids=None, drop_na=False):
        return FactorRDB.readCov(self, table_name, dts=dts, ids=ids, drop_na=drop_na)
    def readFactorCov(self, table_name, dts=None):
        return self.readData(table_name, "FactorCov", dts=dts, ids=None)
    def readSpecificRisk(self, table_name, dts=None, ids=None):
        SpecificRisk = self.readData(table_name, "SpecificRisk", dts=dts, ids=ids)
        if isinstance(SpecificRisk,dict):
            SpecificRisk = pd.DataFrame(SpecificRisk).T
            return SpecificRisk.sort_index()
        else:
            return SpecificRisk
    def readFactorData(self, table_name, dts=None, ids=None):
        return self.readData(table_name, "FactorData", dts=dts, ids=ids)
    def readFactorReturn(self, table_name, dts=None):
        FactorReturn = self.readData(table_name, "FactorReturn", dts=dts, ids=None)
        if isinstance(FactorReturn, dict):
            FactorReturn = pd.DataFrame(FactorReturn).T
            return FactorReturn.sort_index()
        else:
            return FactorReturn
    def readSpecificReturn(self, table_name, dts=None, ids=None):
        SpecificReturn = self.readData(table_name, "SpecificReturn", dts=dts, ids=ids)
        if isinstance(SpecificReturn, dict):
            SpecificReturn = pd.DataFrame(SpecificReturn).T
            return SpecificReturn.sort_index()
        else:
            return SpecificReturn
    def readData(self, table_name, file_name, dts=None, ids=None):
        with self._DataLock:
            with shelve.open(self.MainDir+os.sep+table_name+os.sep+file_name) as DataFile:
                if (dts is not None) and (not isinstance(dts,list)):
                    Data = DataFile.get(dts.strftime("%Y-%m-%d %H:%M:%S.%f"), None)
                    if (Data is not None) and (ids is not None):
                        Data = Data.loc[ids]
                else:
                    if dts is None:
                        DTStrs = list(DataFile)
                    else:
                        DTStrs = {iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts}.intersection(set(DataFile))
                    if ids is None:
                        Data = {dt.datetime.strptime(iDT, "%Y-%m-%d %H:%M:%S.%f"):DataFile[iDT] for iDT in DTStrs}
                    else:
                        Data = {dt.datetime.strptime(iDT, "%Y-%m-%d %H:%M:%S.%f"):DataFile[iDT].loc[ids] for iDT in DTStrs}
                    if Data=={}:
                        Data = None
        return Data
    def writeData(self, table_name, idt, factor_ret=None, specific_ret=None, factor_data=None, factor_cov=None, specific_risk=None, file_value={}):
        TablePath = self.MainDir+os.sep+table_name
        iDTStr = idt.strftime("%Y-%m-%d %H:%M:%S.%f")
        with self._DataLock:
            if table_name not in self._TableDT:
                self._TableDT[table_name] = []
                if not os.path.isdir(TablePath):
                    os.mkdir(TablePath)
            if factor_ret is not None:
                with shelve.open(TablePath+os.sep+"FactorReturn") as DataFile:
                    DataFile[iDTStr] = factor_ret
            if specific_ret is not None:
                with shelve.open(TablePath+os.sep+"SpecificReturn") as DataFile:
                    DataFile[iDTStr] = specific_ret
            if factor_data is not None:
                with shelve.open(TablePath+os.sep+"FactorData") as DataFile:
                    DataFile[iDTStr] = factor_data
            if factor_cov is not None:
                with shelve.open(TablePath+os.sep+"FactorCov") as DataFile:
                    DataFile[iDTStr] = factor_cov
            if specific_risk is not None:
                with shelve.open(TablePath+os.sep+"SpecificRisk") as DataFile:
                    DataFile[iDTStr] = specific_risk
                self._TableDT[table_name].append(idt)
                self._TableDT[table_name].sort()
            for iFile, iValue in file_value.items():
                with shelve.open(TablePath+os.sep+iFile) as DataFile:
                    DataFile[iDTStr] = iValue
        return 0
    def genDSDataPatch(self, file_path, tables, start_dt=None, end_dt=None):
        with __QS_CacheLock__:
            PatchDir = __QS_CachePath__+os.sep+genAvailableName("FRDataPatch", listDirDir(__QS_CachePath__))
            os.mkdir(PatchDir)
        PatchDB = ShelveFRDB(sys_args={"主目录":PatchDir})
        PatchDB.connect()
        for iTableName in tables:
            iDTs = self.getFactorReturnDateTime(iTableName, start_dt=start_dt, end_dt=end_dt)
            iFactorData = self.readData(iTableName, "FactorData", dts=iDTs)
            iFactorReturn = self.readData(iTableName, "FactorReturn", dts=iDTs)
            iSpecificReturn = self.readData(iTableName, "SpecificReturn", dts=iDTs)
            iFactorCov = self.readData(iTableName, "FactorCov", dts=iDTs)
            iSpecificRisk = self.readData(iTableName, "SpecificRisk", dts=iDTs)
            for jDT in iDTs:
                PatchDB.writeData(iTableName, jDT, factor_ret=iFactorReturn.get(jDT, None), specific_ret=iSpecificReturn.get(jDT, None), 
                                 factor_data=iFactorData.get(jDT, None), factor_cov=iFactorCov.get(jDT, None), specific_risk=iSpecificRisk.get(jDT, None))
        CWD = os.getcwd()
        os.chdir(PatchDir)
        ZipFile = zipfile.ZipFile(file_path, mode='a', compression=zipfile.ZIP_DEFLATED)
        for iTableName in tables:
            if os.path.isdir("."+os.sep+iTableName):
                iFiles = os.listdir("."+os.sep+iTableName)
            else:
                continue
            for jFile in iFiles:
                ZipFile.write("."+os.sep+iTableName+os.sep+jFile)
        ZipFile.close()
        os.chdir(CWD)
        shutil.rmtree(PatchDir, ignore_errors=True)
        return 0
    def updateDSData(self, zip_file_path):
        ZipFile = zipfile.ZipFile(zip_file_path, 'r')
        with __QS_CacheLock__:
            PatchDir = __QS_CachePath__+os.sep+genAvailableName("FRDataPatch", listDirDir(__QS_CachePath__))
            os.mkdir(PatchDir)
        for iFile in ZipFile.namelist():
            ZipFile.extract(iFile, PatchDir+os.sep)
        ZipFile.close()
        PatchDB = ShelveFRDB(sys_args={"主目录":PatchDir})
        PatchDB.connect()
        for iTableName in PatchDB._TableDT:
            iDTs = PatchDB.getFactorReturnDateTime(iTableName)
            iFactorData = PatchDB.readData(iTableName, "FactorData", dts=iDTs)
            iFactorReturn = PatchDB.readData(iTableName, "FactorReturn", dts=iDTs)
            iSpecificReturn = PatchDB.readData(iTableName, "SpecificReturn", dts=iDTs)
            iFactorCov = PatchDB.readData(iTableName, "FactorCov", dts=iDTs)
            iSpecificRisk = PatchDB.readData(iTableName, "SpecificRisk", dts=iDTs)
            for jDT in iDTs:
                self.writeData(iTableName, jDT, factor_ret=iFactorReturn.get(jDT, None), specific_ret=iSpecificReturn.get(jDT, None), 
                              factor_data=iFactorData.get(jDT, None), factor_cov=iFactorCov.get(jDT, None), specific_risk=iSpecificRisk.get(jDT, None))                
        shutil.rmtree(PatchDir, ignore_errors=True)
        return 0
