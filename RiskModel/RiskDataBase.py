# coding=utf-8
import os
import shelve
import shutil
from multiprocessing import Lock
import zipfile

import numpy as np
import pandas as pd

from QuantStudio.Tools.AuxiliaryFun import genAvailableName
from QuantStudio.Tools.FileFun import listDirDir,readJSONFile
from QuantStudio.Tools.DateTimeFun import cutDate
from .RiskModelFun import dropRiskMatrixNA, decomposeCov2Corr
from QuantStudio import __QS_Object__, __QS_Error__

# 风险数据库基类, 必须存储的数据有:
# 风险矩阵: Cov, DataFrame(data=协方差, index=[ID], columns=[ID])
class RiskDataBase(__QS_Object__):
    """RiskDataBase"""
    def __init__(self, qs_env, sys_args={}):
        self.QSEnv = qs_env
        self.DBType = "RDB"
        super().__init__(sys_args)
        self.TableDate = {}#{表名：[日期]}
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
    # 获取Shelve数据库中的表名,ignore:忽略的表名列表
    def getTableName(self,ignore=[]):
        return [iTable for iTable in self.TableDate if iTable not in ignore]
    # 获取一个可用的表名
    def genAvailableTableName(self,header='NewRiskData'):
        return genAvailableName(header, self.TableDate)
    # 检查表是否存在
    def checkTableExistence(self,table_name):
        return (table_name in self.TableDate)
    # 获取表的描述信息
    def getTableDescription(self,table_name):
        return ""
    # 设置表的描述信息
    def setTableDescription(self,table_name,description=""):
        return 0
    # 创建表,dates:[日期],前提是当前不存在表table_name, 否则行为不可预知. 
    def createTable(self,table_name,dates):
        self.TableDate[table_name] = dates
        return 0
    # 重命名表
    def renameTable(self,old_table_name,new_table_name):
        self.TableDate[new_table_name] = self.TableDate.pop(old_table_name)
        return 0
    # 删除表
    def deleteTable(self,table_name):
        if not self.checkTableExistence(table_name):
            return 0
        self.TableDate.pop(table_name)
        return 0
    # 获取这张表中的日期序列
    def getTableDate(self,table_name,start_date=None,end_date=None):
        return cutDate(self.TableDate[table_name],start_date,end_date)
    # 获取这张表中的 ID 序列
    def getTableID(self,table_name,idt=None):
        if idt is None:
            idt = self.TableDate[table_name][-1]
        Cov = self.loadCov(table_name,dates=idt)
        return list(Cov.index)
    # 复制表
    def copyTable(self,table_name,new_table_name):
        self.TableDate[new_table_name] = self.TableDate[table_name]
        return 0
    # --------------------------------日期管理-----------------------------------
    # 删除一张表中的某些日期
    def deleteDate(self,table_name,dates):
        for iDate in dates:
            self.TableDate[table_name].remove(iDate)
        return 0
    # 日期变换
    def changeDate(self,table_name,date_change_fun=None,dates=None):
        OldDates = self.TableDate[table_name]
        if dates is not None:
            Dates = list(set(OldDates).intersection(set(dates)))
            Dates.sort()
        else:
            Dates = OldDates
        if date_change_fun is not None:
            Dates = date_change_fun(Dates)
        DeleteDates = list(set(OldDates).difference(set(Dates)))
        DeleteDates.sort()
        self.deleteDate(table_name,DeleteDates)
        return 0
    # ------------------------数据读取--------------------------------------
    # 加载风险矩阵数据到内存中
    def loadCov(self,table_name,dates=None,ids=None,drop_na=False):
        return None
    # 加载相关系数矩阵数据到内存中
    def loadCorr(self,table_name,dates=None,ids=None,drop_na=False):
        Cov = self.loadCov(table_name, dates=dates, ids=ids, drop_na=drop_na)
        if Cov is None:
            return None
        if not isinstance(Cov,dict):
            Corr,_ = decomposeCov2Corr(Cov.values)
            return pd.DataFrame(Corr,index=Cov.index,columns=Cov.columns)
        Corr = {}
        for iDate,iCov in Cov.items():
            iCorr,_ = decomposeCov2Corr(iCov.values)
            Corr[iDate] = pd.DataFrame(iCorr,index=iCov.index,columns=iCov.columns)
        return Corr
    # ------------------------数据存储--------------------------------------
    # 存储数据
    def saveData(self,table_name,idt,cov=None):
        return 0
    # 生成数据补丁
    def genDSDataPatch(self,file_path,tables,start_date=None,end_date=None):
        return 0
    # 用压缩补丁文件更新数据
    def updateDSData(self,zip_file_path):
        return 0
# 建立于 shelve 模块上的风险数据库, 以硬盘中的 shelve 文件作为数据存储方式.
# 每个 shelve 文件以日期为索引, 每个日期对应相应的数据
class ShelveRDB(RiskDataBase):
    """RDB"""
    def __init__(self, qs_env, sys_args={}):
        super().__init__(qs_env, sys_args)
        self.DataLock = Lock()
        if self.QSEnv.SysArgs["OSType"]=="Windows":
            self.Suffix = "dat"
        else:
            self.Suffix = ""
        if not os.path.isdir(self._SysArgs["主目录"]):
            DefaultDir = self.QSEnv.SysArgs["MainPath"]+os.sep+"RiskData"
            if not os.path.isdir(DefaultDir):
                os.mkdir(DefaultDir)
            self._SysArgs["主目录"] = DefaultDir
        return
    def __QS_genSysArgs__(self, args=None, **kwargs):
        if args is not None:
            return args
        SysArgs = {"主目录": self.QSEnv.SysArgs["MainPath"]+os.sep+"RiskData",
                   "_ConfigFilePath":self.QSEnv.SysArgs['LibPath']+os.sep+"RDBConfig.json"}
        ArgInfo = {"主目录":{"type":"Path", "order":0, "refresh":True, "operation":"Directory"}}
        ArgInfo["_ConfigFilePath"] = {"type":"Path", "order":1, "readonly":True, "operation":"Open", "filter":"Configure files (*.json)"}
        return QSArgs(args=SysArgs, arg_info=ArgInfo, callback=self.__QS_onSysArgChanged__)
    def __QS_onSysArgChanged__(self, change_type, change_info):
        Args, Key, Value = change_info
        if (Args is self._SysArgs) and (Key=="主目录"):
            if self.SysArgs["主目录"]!=Value:
                OldValue,self.SysArgs["主目录"] = self.SysArgs["主目录"],Value
                if self._isAvailable:
                    try:
                        self.connect()
                    except Exception as e:
                        self.SysArgs["主目录"] = OldValue
                        raise e
            return True
        else:
            return super().__QS_onSysArgChanged__(change_type, change_info)
    def __setattr__(self, key, value):
        if (key=="_SysArgs") and hasattr(self,"_SysArgs"):
            OldValue = self._SysArgs["主目录"]
            super().__setattr__(key, value)
            if self._SysArgs.get("主目录",None)!=OldValue:
                if self._isAvailable():
                    try:
                        self.connect()
                    except Exception as e:
                        self._SysArgs._QS_MonitorChange = False
                        self._SysArgs["主目录"] = OldValue
                        self._SysArgs._QS_MonitorChange = True
                        raise e
        else:
            return super().__setattr__(key,value)
    # 链接数据库
    def connect(self):
        if not os.path.isdir(self._SysArgs["主目录"]):
            raise QSError("不存在 ShelveRDB 的主目录: %s!" % self._SysArgs["主目录"])
        AllTables = listDirDir(self._SysArgs["主目录"])
        self.TableDate = {}#{表名：[日期]}
        self.DataLock.acquire()
        for iTable in AllTables:
            iTablePath = self._SysArgs["主目录"]+os.sep+iTable
            with shelve.open(iTablePath+os.sep+"__TableInfo") as iTableInfoFile:
                iType = iTableInfoFile.get("DBType")
                if iType is None:
                    if os.path.isfile(iTablePath+os.sep+"Cov"+("."+self.Suffix if self.Suffix!="" else "")):
                        iType = "RDB"
                    else:
                        continue
                    iTableInfoFile["DBType"] = iType
            if iType=="RDB":
                with shelve.open(iTablePath+os.sep+"Cov") as iFile:
                    iDates = list(iFile.keys())
                iDates.sort()
                self.TableDate[iTable] = iDates
        self.DataLock.release()
        self._isAvailable = True
        return 0
    # -------------------------------表的管理---------------------------------
    # 获取表的描述信息
    def getTableDescription(self,table_name):
        self.DataLock.acquire()
        with shelve.open(self._SysArgs["主目录"]+os.sep+table_name+os.sep+"__TableInfo") as TableInfoFile:
            FieldData = TableInfoFile.get("TableDescription","")
        self.DataLock.release()
        return FieldData
    # 设置表的描述信息
    def setTableDescription(self,table_name,description=""):
        self.DataLock.acquire()
        with shelve.open(self._SysArgs["主目录"]+os.sep+table_name+os.sep+"__TableInfo") as TableInfoFile:
            TableInfoFile["TableDescription"] = description
        self.DataLock.release()
        return 0
    # 创建表,dates:[日期],前提是当前不存在表table_name, 否则行为不可预知. 
    def createTable(self,table_name,dates):
        self.TableDate[table_name] = dates
        self.DataLock.acquire()
        os.mkdir(self._SysArgs["主目录"]+os.sep+table_name)
        self.DataLock.release()
        return 0
    # 重命名表
    def renameTable(self,old_table_name,new_table_name):
        self.TableDate[new_table_name] = self.TableDate.pop(old_table_name)
        self.DataLock.acquire()
        os.rename(self._SysArgs["主目录"]+os.sep+old_table_name,self._SysArgs["主目录"]+os.sep+new_table_name)
        self.DataLock.release()
        return 0
    # 删除表
    def deleteTable(self,table_name):
        if not self.checkTableExistence(table_name):
            return 0
        self.TableDate.pop(table_name)
        if not os.path.isdir(self._SysArgs["主目录"]+os.sep+table_name):
            return 0
        self.DataLock.acquire()
        shutil.rmtree(self._SysArgs["主目录"]+os.sep+table_name,ignore_errors=True)
        self.DataLock.release()
        return 0
    # 复制表
    def copyTable(self,table_name,new_table_name):
        FactorRDB.copyTable(self,table_name,new_table_name)
        SrcDirPath = self._SysArgs["主目录"]+os.sep+table_name
        DstDirPath = self._SysArgs["主目录"]+os.sep+new_table_name
        shutil.copytree(SrcDirPath, DstDirPath)
        return 0
    # --------------------------------日期管理-----------------------------------
    # 删除一张表中的某些日期
    def deleteDate(self,table_name,dates):
        self.DataLock.acquire()
        with shelve.open(self._SysArgs["主目录"]+os.sep+table_name+os.sep+"Cov") as CovFile:
            for iDate in dates:
                self.TableDate[table_name].remove(iDate)
                CovFile.pop(iDate)
        self.DataLock.release()
        return 0
    # ------------------------数据读取--------------------------------------
    # 加载风险矩阵数据到内存中
    def loadCov(self,table_name,dates=None,ids=None,drop_na=False):
        self.DataLock.acquire()
        with shelve.open(self._SysArgs["主目录"]+os.sep+table_name+os.sep+"Cov") as DataFile:
            if (dates is not None) and (not isinstance(dates,list)):
                Data = DataFile.get(dates)
                if (Data is not None) and (ids is not None):
                    Data = Data.ix[ids,ids]
                if (Data is not None) and drop_na:
                    Data = dropRiskMatrixNA(Data)
            else:
                if dates is None:
                    dates = list(DataFile.keys())
                else:
                    dates = set(dates).intersection(set(DataFile.keys()))
                if not drop_na:
                    dropRiskMatrixNA = lambda x:x
                if ids is None:
                    Data = {iDate:dropRiskMatrixNA(DataFile[iDate]) for iDate in dates}
                else:
                    Data = {iDate:dropRiskMatrixNA(DataFile[iDate].ix[ids,ids]) for iDate in dates}
                if Data=={}:
                    Data = None
        self.DataLock.release()
        return Data
    # ------------------------数据存储--------------------------------------
    # 存储数据
    def saveData(self,table_name,idt,cov=None,file_value={}):
        TablePath = self._SysArgs["主目录"]+os.sep+table_name
        self.DataLock.acquire()
        if table_name not in self.TableDate:
            self.TableDate[table_name] = []
            if not os.path.isdir(TablePath):
                os.mkdir(TablePath)
        if cov is not None:
            with shelve.open(TablePath+os.sep+"Cov") as DataFile:
                DataFile[idt] = cov
                self.TableDate[table_name] = list(DataFile.keys())
            self.TableDate[table_name].sort()
        for iFile,iValue in file_value.items():
            with shelve.open(TablePath+os.sep+iFile) as DataFile:
                DataFile[idt] = iValue
        self.DataLock.release()
        return 0
    # 生成数据补丁
    def genDSDataPatch(self,file_path,tables,start_date=None,end_date=None):
        PatchDir,DirLock = self.QSEnv.createPrivateCacheDir(dir_name="RDataPatch")
        PatchDB = ShelveRDB(PatchDir,DirLock,self.QSEnv)
        PatchDB.connect()
        for iTableName in tables:
            iDates = self.getTableDate(iTableName,start_date=start_date,end_date=end_date)
            iCov = self.loadCov(iTableName,dates=iDates)
            for jDate in iDates:
                PatchDB.saveData(iTableName,jDate,cov=iCov.get(jDate))
        CWD = os.getcwd()
        os.chdir(PatchDir)
        ZipFile = zipfile.ZipFile(file_path,mode='a',compression=zipfile.ZIP_DEFLATED)
        for iTableName in tables:
            if os.path.isdir("."+os.sep+iTableName):
                iFiles = os.listdir("."+os.sep+iTableName)
            else:
                continue
            for jFile in iFiles:
                ZipFile.write("."+os.sep+iTableName+os.sep+jFile+("."+self.Suffix if self.Suffix!="" else ""))
        ZipFile.close()
        os.chdir(CWD)
        shutil.rmtree(PatchDir,ignore_errors=True)
        return 0
    # 用压缩补丁文件更新数据
    def updateDSData(self,zip_file_path):
        ZipFile = zipfile.ZipFile(zip_file_path, 'r')
        PatchDir,DirLock = self.QSEnv.createPrivateCacheDir(dir_name="RDataPatch")
        for iFile in ZipFile.namelist():
            ZipFile.extract(iFile,PatchDir+os.sep)
        ZipFile.close()
        PatchDB = ShelveRDB(PatchDir,DirLock,self.QSEnv)
        PatchDB.connect()
        for iTableName in PatchDB.TableDate:
            iDates = PatchDB.getTableDate(iTableName,start_date=start_date,end_date=end_date)
            iCov = PatchDB.loadCov(iTableName,dates=iDates)
            for jDate in iDates:
                self.saveData(iTableName,jDate,cov=iCov.get(jDate))
        shutil.rmtree(PatchDir,ignore_errors=True)
        return 0

# 多因子风险数据库基类, 即风险矩阵可以分解成 V=X*F*X'+D 的模型, 其中 D 是对角矩阵, 必须存储的数据有:
# 因子风险矩阵: FactorCov(F), DataFrame(data=协方差, index=[因子], columns=[因子])
# 特异性风险: SpecificRisk(D), Series(data=方差, index=[ID])
# 因子截面数据: FactorData(X), DataFrame(data=因子数据, index=[ID], columns=[因子])
# 可选存储的数据有:
# 因子收益率: FactorReturn, Series(data=收益率, index=[因子])
# 特异性收益率: SpecificReturn, Series(data=收益率, index=[ID])
# 回归统计量: Statistics, {"tValue":Series(data=统计量, index=[因子]),"FValue":double,"rSquared":double,"rSquared_Adj":double}
class FactorRDB(RiskDataBase):
    """FactorRDB"""
    def __init__(self, qs_env, sys_args={}):
        super().__init__(qs_env, sys_args)
        self.DBType = "FRDB"
        return
    # -------------------------------表的管理---------------------------------
    # 获取表的所有因子
    def getTableFactor(self,table_name):
        return []
    # 获取这张表中的 ID 序列
    def getTableID(self,table_name,idt=None):
        if idt is None:
            idt = self.TableDate[table_name][-1]
        SpecificRisk = self.loadSpecificRisk(table_name,dates=idt)
        return list(SpecificRisk.index)
    # --------------------------------日期管理-----------------------------------
    # 调整日期序列
    def _adjustDates(self,dates,start_date=None,end_date=None,is_sorted=True):
        Dates = cutDate(dates,start_date,end_date)
        if is_sorted:
            Dates.sort()
        return Dates
    # 获取因子收益的日期
    def getFactorReturnDate(self,table_name,start_date=None,end_date=None,is_sorted=True):
        FactorReturn = self.loadFactorReturn(table_name)
        if FactorReturn is not None:
            return self._adjustDates(FactorReturn.index,start_date=start_date,end_date=end_date,is_sorted=is_sorted)
        else:
            return []
    # 获取特异性收益的日期
    def getSpecificReturnDate(self,table_name,start_date=None,end_date=None,is_sorted=True):
        SpecificReturn = self.loadSpecificRisk(table_name)
        if SpecificReturn is not None:
            return self._adjustDates(SpecificReturn.index,start_date=start_date,end_date=end_date,is_sorted=is_sorted)
        else:
            return []
    # ------------------------数据读取--------------------------------------
    # 加载风险矩阵数据到内存中
    def loadCov(self,table_name,dates=None,ids=None,drop_na=False):
        FactorCov = self.loadFactorCov(table_name,dates=dates)
        FactorData = self.loadFactorData(table_name,dates=dates,ids=ids)
        SpecificRisk = self.loadSpecificRisk(table_name,dates=dates,ids=ids)
        if (dates is not None) and (not isinstance(dates,list)):
            if ids is None:
                ids = SpecificRisk.index
                FactorData = FactorData.ix[ids]
            CovMatrix = np.dot(np.dot(FactorData.values,FactorCov.values),FactorData.values.T)+np.diag(SpecificRisk.values**2)
            CovMatrix = pd.DataFrame(CovMatrix,index=ids,columns=ids)
            if drop_na:
                return dropRiskMatrixNA(CovMatrix)
            return CovMatrix
        Data = {}
        for iDate in FactorCov:
            if ids is None:
                iIDs = SpecificRisk.loc[iDate].index
                iFactorData = FactorData[iDate].ix[iIDs]
            else:
                iIDs = ids
                iFactorData = FactorData[iDate]
            iCovMatrix = np.dot(np.dot(iFactorData.values,FactorCov[iDate].values),iFactorData.values.T)+np.diag(SpecificRisk.loc[iDate].values**2)
            iCovMatrix = pd.DataFrame(iCovMatrix,index=iIDs,columns=iIDs)
            if drop_na:
                iCovMatrix = dropRiskMatrixNA(iCovMatrix)
            Data[iDate] = iCovMatrix
        return Data
    # 加载因子风险矩阵数据到内存中
    def loadFactorCov(self,table_name,dates=None):
        return None
    # 加载特异性风险数据到内存中
    def loadSpecificRisk(self,table_name,dates=None,ids=None):
        return None
    # 加载截面数据到内存中
    def loadFactorData(self,table_name,dates=None,ids=None):
        return None
    # 加载因子收益率数据到内存中
    def loadFactorReturn(self,table_name,dates=None):
        return None
    # 加载残余收益率数据到内存中
    def loadSpecificReturn(self,table_name,dates=None,ids=None):
        return None
    # ------------------------数据存储--------------------------------------
    # 存储数据
    def saveData(self,table_name,idt,factor_ret=None,specific_ret=None,factor_data=None,factor_cov=None,specific_risk=None):
        return 0

# 建立于 shelve 模块上的多因子风险数据库, 以硬盘中的 shelve 文件作为数据存储方式.
# 每个 shelve 文件以日期为索引, 每个日期对应相应的数据
class ShelveFRDB(ShelveRDB,FactorRDB):
    """FRDB"""
    def __init__(self, qs_env, sys_args={}):
        super().__init__(qs_env, sys_args)
        self.DBType = 'FRDB'
        return
    def __QS_genSysArgs__(self, args=None, **kwargs):
        if args is None:
            SysArgs = super().__QS_genSysArgs__(args=args, **kwargs)
            SysArgs._QS_MonitorChange = False
            SysArgs["_ConfigFilePath"] = self.QSEnv.SysArgs['LibPath']+os.sep+"FRDBConfig.json"
            SysArgs._QS_MonitorChange = True
            return SysArgs
        return args
    # 更新信息
    def connect(self):
        if not os.path.isdir(self._SysArgs["主目录"]):
            raise QSError("不存在 ShelveFRDB 的主目录: %s!" % self._SysArgs["主目录"])
        AllTables = listDirDir(self._SysArgs["主目录"])
        self.TableDate = {}#{表名：[日期]}
        self.DataLock.acquire()
        for iTable in AllTables:
            iTablePath = self._SysArgs["主目录"]+os.sep+iTable
            with shelve.open(iTablePath+os.sep+"__TableInfo") as iTableInfoFile:
                iType = iTableInfoFile.get("DBType")
                if iType is None:
                    if os.path.isfile(iTablePath+os.sep+"FactorCov"+("."+self.Suffix if self.Suffix!="" else "")):
                        iType = "FRDB"
                    else:
                        continue
                    iTableInfoFile["DBType"] = iType
            if iType=="FRDB":
                with shelve.open(iTablePath+os.sep+"SpecificRisk") as iFile:
                    iDates = list(iFile.keys())
                iDates.sort()
                self.TableDate[iTable] = iDates
        self.DataLock.release()
        self._isAvailable = True
        return 0
    # -------------------------------表的管理---------------------------------
    # 获取表的所有因子
    def getTableFactor(self,table_name):
        self.DataLock.acquire()
        with shelve.open(self._SysArgs["主目录"]+os.sep+table_name+os.sep+"FactorCov") as DataFile:
            FactorCov = DataFile[self.TableDate[table_name][-1]]
        self.DataLock.release()
        return list(FactorCov.index)
    # 获取这张表中的 ID 序列
    def getTableID(self,table_name,idt=None):
        return FactorRDB.getTableID(self,table_name,idt=idt)
    # --------------------------------日期管理------------------------------
    # 获取因子收益的日期
    def getFactorReturnDate(self,table_name,start_date=None,end_date=None,is_sorted=True):
        self.DataLock.acquire()
        with shelve.open(self._SysArgs["主目录"]+os.sep+table_name+os.sep+"FactorReturn") as DataFile:
            Dates = list(DataFile.keys())
        self.DataLock.release()
        return self._adjustDates(Dates,start_date=start_date,end_date=end_date,is_sorted=is_sorted)
    # 获取特异性收益的日期
    def getSpecificReturnDate(self,table_name,start_date=None,end_date=None,is_sorted=True):
        self.DataLock.acquire()
        with shelve.open(self._SysArgs["主目录"]+os.sep+table_name+os.sep+"SpecificReturn") as DataFile:
            Dates = list(DataFile.keys())
        self.DataLock.release()
        return self._adjustDates(Dates,start_date=start_date,end_date=end_date,is_sorted=is_sorted)
    # 删除一张表中的某些日期
    def deleteDate(self,table_name,dates):
        self.DataLock.acquire()
        with shelve.open(self._SysArgs["主目录"]+os.sep+table_name+os.sep+"FactorCov") as FactorCovFile:
            with shelve.open(self._SysArgs["主目录"]+os.sep+table_name+os.sep+"SpecificRisk") as SpecificRiskFile:
                for iDate in dates:
                    self.TableDate[table_name].remove(iDate)
                    FactorCovFile.pop(iDate)
                    SpecificRiskFile.pop(iDate)
        self.DataLock.release()
        return 0
    # ------------------------数据读取--------------------------------------
    # 加载风险矩阵数据到内存中
    def loadCov(self,table_name,dates=None,ids=None,drop_na=False):
        return FactorRDB.loadCov(self,table_name,dates=dates,ids=ids,drop_na=drop_na)
    # 加载因子风险矩阵数据到内存中
    def loadFactorCov(self,table_name,dates=None):
        return self.loadData(table_name,"FactorCov",dates=dates,ids=None)
    # 加载特异性风险数据到内存中
    def loadSpecificRisk(self,table_name,dates=None,ids=None):
        SpecificRisk = self.loadData(table_name,"SpecificRisk",dates=dates,ids=ids)
        if isinstance(SpecificRisk,dict):
            SpecificRisk = pd.DataFrame(SpecificRisk).T
            return SpecificRisk.sort_index()
        else:
            return SpecificRisk
    # 加载截面数据到内存中
    def loadFactorData(self,table_name,dates=None,ids=None):
        return self.loadData(table_name,"FactorData",dates=dates,ids=ids)
    # 加载因子收益率数据到内存中
    def loadFactorReturn(self,table_name,dates=None):
        FactorReturn = self.loadData(table_name,"FactorReturn",dates=dates,ids=None)
        if isinstance(FactorReturn,dict):
            FactorReturn = pd.DataFrame(FactorReturn).T
            return FactorReturn.sort_index()
        else:
            return FactorReturn
    # 加载残余收益率数据到内存中
    def loadSpecificReturn(self,table_name,dates=None,ids=None):
        SpecificReturn = self.loadData(table_name,"SpecificReturn",dates=dates,ids=ids)
        if isinstance(SpecificReturn,dict):
            SpecificReturn = pd.DataFrame(SpecificReturn).T
            return SpecificReturn.sort_index()
        else:
            return SpecificReturn
    # 加载数据到内存中
    def loadData(self,table_name,file_name,dates=None,ids=None):
        self.DataLock.acquire()
        with shelve.open(self._SysArgs["主目录"]+os.sep+table_name+os.sep+file_name) as DataFile:
            if (dates is not None) and (not isinstance(dates,list)):
                Data = DataFile.get(dates)
                if (Data is not None) and (ids is not None):
                    Data = Data.ix[ids]
            else:
                if dates is None:
                    dates = list(DataFile.keys())
                else:
                    dates = set(dates).intersection(set(DataFile.keys()))
                if ids is None:
                    Data = {iDate:DataFile[iDate] for iDate in dates}
                else:
                    Data = {iDate:DataFile[iDate].ix[ids] for iDate in dates}
                if Data=={}:
                    Data = None
        self.DataLock.release()
        return Data
    # ------------------------数据存储--------------------------------------
    # 存储数据
    def saveData(self,table_name,idt,factor_ret=None,specific_ret=None,factor_data=None,factor_cov=None,specific_risk=None,file_value={}):
        TablePath = self._SysArgs["主目录"]+os.sep+table_name
        self.DataLock.acquire()
        if table_name not in self.TableDate:
            self.TableDate[table_name] = []
            if not os.path.isdir(TablePath):
                os.mkdir(TablePath)
        if factor_ret is not None:
            with shelve.open(TablePath+os.sep+"FactorReturn") as DataFile:
                DataFile[idt] = factor_ret
        if specific_ret is not None:
            with shelve.open(TablePath+os.sep+"SpecificReturn") as DataFile:
                DataFile[idt] = specific_ret
        if factor_data is not None:
            with shelve.open(TablePath+os.sep+"FactorData") as DataFile:
                DataFile[idt] = factor_data
        if factor_cov is not None:
            with shelve.open(TablePath+os.sep+"FactorCov") as DataFile:
                DataFile[idt] = factor_cov
        if specific_risk is not None:
            with shelve.open(TablePath+os.sep+"SpecificRisk") as DataFile:
                DataFile[idt] = specific_risk
                self.TableDate[table_name] = list(DataFile.keys())
            self.TableDate[table_name].sort()
        for iFile,iValue in file_value.items():
            with shelve.open(TablePath+os.sep+iFile) as DataFile:
                DataFile[idt] = iValue
        self.DataLock.release()
        return 0
    # 生成数据补丁
    def genDSDataPatch(self,file_path,tables,start_date=None,end_date=None):
        PatchDir,DirLock = self.QSEnv.createPrivateCacheDir(dir_name="FRDataPatch")
        PatchDB = ShelveFRDB(PatchDir,DirLock,self.QSEnv)
        PatchDB.connect()
        for iTableName in tables:
            iDates = self.getFactorReturnDate(iTableName,start_date=start_date,end_date=end_date)
            iFactorData = self.loadData(iTableName, "FactorData",dates=iDates)
            iFactorReturn = self.loadData(iTableName,"FactorReturn",dates=iDates)
            iSpecificReturn = self.loadData(iTableName,"SpecificReturn",dates=iDates)
            iFactorCov = self.loadData(iTableName,"FactorCov",dates=iDates)
            iSpecificRisk = self.loadData(iTableName,"SpecificRisk",dates=iDates)
            for jDate in iDates:
                PatchDB.saveData(iTableName,jDate,factor_data=iFactorData.get(jDate))
                PatchDB.saveData(iTableName,jDate,factor_ret=iFactorReturn.get(jDate))
                PatchDB.saveData(iTableName,jDate,specific_ret=iSpecificReturn.get(jDate))
                PatchDB.saveData(iTableName,jDate,factor_cov=iFactorCov.get(jDate))
                PatchDB.saveData(iTableName,jDate,specific_risk=iSpecificRisk.get(jDate))
        CWD = os.getcwd()
        os.chdir(PatchDir)
        ZipFile = zipfile.ZipFile(file_path,mode='a',compression=zipfile.ZIP_DEFLATED)
        for iTableName in tables:
            if os.path.isdir("."+os.sep+iTableName):
                iFiles = os.listdir("."+os.sep+iTableName)
            else:
                continue
            for jFile in iFiles:
                ZipFile.write("."+os.sep+iTableName+os.sep+jFile)
        ZipFile.close()
        os.chdir(CWD)
        shutil.rmtree(PatchDir,ignore_errors=True)
        return 0
    # 用压缩补丁文件更新数据
    def updateDSData(self,zip_file_path):
        ZipFile = zipfile.ZipFile(zip_file_path, 'r')
        PatchDir,DirLock = self.QSEnv.createPrivateCacheDir(dir_name="FRDataPatch")
        for iFile in ZipFile.namelist():
            ZipFile.extract(iFile,PatchDir+os.sep)
        ZipFile.close()
        PatchDB = ShelveFRDB(PatchDir,DirLock,self.QSEnv)
        PatchDB.connect()
        for iTableName in PatchDB.TableDate:
            iDates = PatchDB.getFactorReturnDate(iTableName)
            iFactorData = PatchDB.loadData(iTableName, "FactorData",dates=iDates)
            iFactorReturn = PatchDB.loadData(iTableName,"FactorReturn",dates=iDates)
            iSpecificReturn = PatchDB.loadData(iTableName,"SpecificReturn",dates=iDates)
            iFactorCov = PatchDB.loadData(iTableName,"FactorCov",dates=iDates)
            iSpecificRisk = PatchDB.loadData(iTableName,"SpecificRisk",dates=iDates)
            for jDate in iDates:
                self.saveData(iTableName,jDate,factor_data=iFactorData.get(jDate))
                self.saveData(iTableName,jDate,factor_ret=iFactorReturn.get(jDate))
                self.saveData(iTableName,jDate,specific_ret=iSpecificReturn.get(jDate))
                self.saveData(iTableName,jDate,factor_cov=iFactorCov.get(jDate))
                self.saveData(iTableName,jDate,specific_risk=iSpecificRisk.get(jDate))
        shutil.rmtree(PatchDir,ignore_errors=True)
        return 0
# 多因子风险数据库的 Proxy
class ProxyFRDB(FactorRDB):
    """FRDB"""
    def __init__(self, qs_env=None):
        self.QSEnv = qs_env
        self.TableDate = {}#{表名：[日期]}
        self._isAvailable = False
        self.CallbackRoute = "/Callback/"+self.QSEnv.SysArgs['User']
        self.DBType = 'FRDB'
        return
    # 链接数据库
    def connect(self):
        if not self.QSEnv.SysArgs['isLogin']:
            self.QSEnv.SysArgs['LastErrorMsg'] = "服务器未连接, 因子库连接失败!"
            return 0
        Msg = (["MainQSEnv","FRDB"],"TableDate",None)
        Error,TableDate = self.QSEnv.communicateWithServer(msg=Msg,route=self.CallbackRoute)
        if Error!=1:
            self.TableDate = {}
            self.QSEnv.SysArgs['LastErrorMsg'] = TableDate
            self._isAvailable = False
            return Error
        self.TableDate = TableDate
        self._isAvailable = True
        return 0
    # -------------------------------表的管理---------------------------------
    # 获取表的描述信息
    def getTableDescription(self,table_name):
        Msg = (["MainQSEnv","FRDB"],"getTableDescription",{"table_name":table_name})
        Error,Rslt = self.QSEnv.communicateWithServer(msg=Msg,route=self.CallbackRoute)
        if Error!=1:
            self.QSEnv.SysArgs['LastErrorMsg'] = Rslt
            return None
        return Rslt
    def DefaultRejectFun(self):
        self.QSEnv.SysArgs["LastErrorMsg"] = "没有权限修改风险数据"
        return 0
    # 设置表的描述信息
    def setTableDescription(self,table_name,description=""):
        return self.DefaultRejectFun()
    # 创建表,dates:[日期],前提是当前不存在表table_name, 否则行为不可预知. 
    def createTable(self,table_name,dates):
        return self.DefaultRejectFun()
    # 重命名表
    def renameTable(self,old_table_name,new_table_name):
        return self.DefaultRejectFun()
    # 删除表
    def deleteTable(self,table_name):
        return self.DefaultRejectFun()
    # 获取表的所有因子
    def getTableFactor(self,table_name):
        Msg = (["MainQSEnv","FRDB"],"getTableFactor",{"table_name":table_name})
        Error,Rslt = self.QSEnv.communicateWithServer(msg=Msg,route=self.CallbackRoute)
        if Error!=1:
            self.QSEnv.SysArgs['LastErrorMsg'] = Rslt
            return None
        return Rslt
    # --------------------------------日期管理-----------------------------------
    # 获取因子收益的日期
    def getFactorReturnDate(self,table_name,start_date=None,end_date=None,is_sorted=True):
        Msg = (["MainQSEnv","FRDB"],"getFactorReturnDate",{"table_name":table_name,"start_date":start_date,"end_date":end_date,"is_sorted":is_sorted})
        Error,Rslt = self.QSEnv.communicateWithServer(msg=Msg,route=self.CallbackRoute)
        if Error!=1:
            self.QSEnv.SysArgs['LastErrorMsg'] = Rslt
            return None
        return Rslt
    # 获取特异性收益的日期
    def getSpecificReturnDate(self,table_name,start_date=None,end_date=None,is_sorted=True):
        Msg = (["MainQSEnv","FRDB"],"getSpecificReturnDate",{"table_name":table_name,"start_date":start_date,"end_date":end_date,"is_sorted":is_sorted})
        Error,Rslt = self.QSEnv.communicateWithServer(msg=Msg,route=self.CallbackRoute)
        if Error!=1:
            self.QSEnv.SysArgs['LastErrorMsg'] = Rslt
            return None
        return Rslt
    # 删除一张表中的某些日期
    def deleteDate(self,table_name,dates):
        return self.DefaultRejectFun()
    # 日期变换
    def changeDate(self,table_name,date_change_fun=None,dates=None):
        return self.DefaultRejectFun()
    # ------------------------数据读取--------------------------------------
    # 加载Shelve数据库中的风险数据到内存中
    def loadFactorCov(self,table_name,dates=None):
        Msg = (["MainQSEnv","FRDB"],"loadFactorCov",{"table_name":table_name,"dates":dates})
        Error,Rslt = self.QSEnv.communicateWithServer(msg=Msg,route=self.CallbackRoute)
        if Error!=1:
            self.QSEnv.SysArgs['LastErrorMsg'] = Rslt
            return None
        return Rslt
    # 加载个股风险到内存中
    def loadSpecificRisk(self,table_name,dates=None,ids=None):
        Msg = (["MainQSEnv","FRDB"],"loadSpecificRisk",{"table_name":table_name,"dates":dates,"ids":ids})
        Error,Rslt = self.QSEnv.communicateWithServer(msg=Msg,route=self.CallbackRoute)
        if Error!=1:
            self.QSEnv.SysArgs['LastErrorMsg'] = Rslt
            return None
        return Rslt
    # 加载截面数据到内存中
    def loadFactorData(self,table_name,dates=None,ids=None):
        Msg = (["MainQSEnv","FRDB"],"loadFactorData",{"table_name":table_name,"dates":dates,"ids":ids})
        Error,Rslt = self.QSEnv.communicateWithServer(msg=Msg,route=self.CallbackRoute)
        if Error!=1:
            self.QSEnv.SysArgs['LastErrorMsg'] = Rslt
            return None
        return Rslt
    # 加载因子收益率数据到内存中
    def loadFactorReturn(self,table_name,dates=None):
        Msg = (["MainQSEnv","FRDB"],"loadFactorReturn",{"table_name":table_name,"dates":dates})
        Error,Rslt = self.QSEnv.communicateWithServer(msg=Msg,route=self.CallbackRoute)
        if Error!=1:
            self.QSEnv.SysArgs['LastErrorMsg'] = Rslt
            return None
        return Rslt
    # 加载残余收益率数据到内存中
    def loadSpecificReturn(self,table_name,dates=None,ids=None):
        Msg = (["MainQSEnv","FRDB"],"loadSpecificReturn",{"table_name":table_name,"dates":dates,"ids":ids})
        Error,Rslt = self.QSEnv.communicateWithServer(msg=Msg,route=self.CallbackRoute)
        if Error!=1:
            self.QSEnv.SysArgs['LastErrorMsg'] = Rslt
            return None
        return Rslt
    # 存储数据
    def saveData(self,table_name,idt,factor_ret=None,specific_ret=None,factor_data=None,factor_cov=None,specific_risk=None):
        return self.DefaultRejectFun()