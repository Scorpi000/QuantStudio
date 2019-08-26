# coding=utf-8
"""文件系统操作以及文件读写函数"""
import os
import io
import shutil
import json
import datetime as dt
import shelve
import tempfile
from functools import lru_cache

import numpy as np
import pandas as pd

from QuantStudio import __QS_Error__
from QuantStudio.Tools.AuxiliaryFun import genAvailableName
from QuantStudio.Tools.DataTypeConversionFun import Series2DataFrame

# 复制文件夹到指定位置，如果该文件夹已经存在，则根据if_exist参数进行操作,replace表示覆盖原文件，skip表示保留原文件
def copyDir(source_dir,target_pos,if_exist='replace'):
    DirName = source_dir.split(os.sep)[-1]
    if DirName not in os.listdir(target_pos):
        os.mkdir(target_pos+os.sep+DirName)
    elif if_exist=='skip':
        return 0
    FileList = os.listdir(source_dir)
    for iFile in FileList:
        if os.path.isdir(source_dir+os.sep+iFile):
            copyDir(source_dir+os.sep+iFile, target_pos+os.sep+DirName,if_exist)
        elif (if_exist=='replace') or (iFile not in os.listdir(target_pos+os.sep+DirName)):
            shutil.copy(source_dir+os.sep+iFile, target_pos+os.sep+DirName+os.sep+iFile)
    return 0
# 清空目录下的内容
def clearDir(dir_path):
    AllFileNames = os.listdir(path=dir_path)
    nFailed = 0
    for iFileName in AllFileNames:
        iFilePath = dir_path+os.sep+iFileName
        if os.path.isdir(iFilePath):
            try:
                shutil.rmtree(iFilePath)
            except:
                nFailed += 1
        elif os.path.isfile(iFilePath):
            try:
                os.remove(iFilePath)
            except:
                nFailed += 1
    return nFailed
# 获取一个目录下所有的文件夹名称列表
def listDirDir(dir_path='.'):
    AllFileNames = os.listdir(path=dir_path)
    Rslt = []
    for iFileName in AllFileNames:
        if os.path.isdir(dir_path+os.sep+iFileName):
            Rslt.append(iFileName)
    return Rslt
# 获取一个目录下给定后缀的文件名称列表
def listDirFile(dir_path='.',suffix='csv'):
    suffix = suffix.lower()
    AllFileNames = os.listdir(path=dir_path)
    if suffix is None:
        return [iFileName for iFileName in AllFileNames if not os.path.isdir(dir_path+os.sep+iFileName)]
    else:
        Rslt = []
        if suffix=="":
            for iFileName in AllFileNames:
                iFileNameList = iFileName.split('.')
                if (len(iFileNameList)==1) and (not os.path.isdir(dir_path+os.sep+iFileName)):
                    Rslt.append(iFileName)
        else:
            for iFileName in AllFileNames:
                iFileNameList = iFileName.split('.')
                if (len(iFileNameList)>1) and (iFileNameList[-1].lower()==suffix):
                    Rslt.append('.'.join(iFileNameList[:-1]))
        return Rslt
# 遍历指定文件夹下的给定后缀的文件路径，可选是否遍历子文件夹,如果后缀名为None，遍历所有文件（不包括文件夹），后缀名为-1,遍历所有文件（包括文件夹）,后缀名为-2，遍历所有文件夹
def traverseDir(dir_path='.',suffix=None,traverse_subdir=True):
    if isinstance(suffix,str):
        suffix = suffix.lower()
    AllFileNames = os.listdir(path=dir_path)
    for iFileName in AllFileNames:
        iFilePath = dir_path+os.sep+iFileName
        if os.path.isdir(iFilePath):# 该元素是文件夹
            if isinstance(suffix,int) and (suffix<0):
                yield iFilePath
            if traverse_subdir:
                for jSubFilePath in traverseDir(dir_path=iFilePath,suffix=suffix,traverse_subdir=traverse_subdir):
                    yield jSubFilePath
            continue
        else:# 该元素是文件
            if suffix is None:
                yield iFilePath
            elif isinstance(suffix,int) and (suffix==-1):
                yield iFilePath
            elif isinstance(suffix,str):
                iFileNameList = iFileName.split('.')
                if ((suffix=='') and (len(iFileNameList)==1)) or ((len(iFileNameList)>1) and (iFileNameList[-1].lower()==suffix)):
                    yield iFilePath
                else:
                    continue
            else:
                continue
# 删除shelve文件
def deleteShelveFile(file_path):
    if os.path.isfile(file_path+".dat"):
        os.remove(file_path+".dat")
    if os.path.isfile(file_path+".bak"):
        os.remove(file_path+".bak")
    if os.path.isfile(file_path+".dir"):
        os.remove(file_path+".dir")
    return 0
# 输出字典序列到csv文件
def writeDictSeries2CSV(dict_series,file_path):
    Index = list(dict_series.index)
    Index.sort()
    nInd = len(Index)
    nLen = 0
    Lens = [len(dict_series[iInd]) for iInd in Index]
    nLen = max(Lens)
    DataArray = np.array([('',)*nInd*2]*(nLen+1),dtype='O')
    for i,iInd in enumerate(Index):
        DataArray[:Lens[i]+1,2*i:2*i+2] = np.array([(iInd,'')]+list(dict_series[iInd].items()))
    np.savetxt(file_path,DataArray,fmt='%s',delimiter=',')
    return 0
# 将函数定义写入文件,file_path:文件路径名;operator_info：算子定义信息，{'算子名称':'','算子定义':'','算子输入',['',''],'导入模块':[[],[]]}, to_truncate:是否清空文件
def writeFun2File(file_path,operator_info,to_truncate=True):
    Modules = {}# {父模块名:[(子模块名,模块别称)]}
    for i in range(len(operator_info['导入模块'][0])):
        if operator_info['导入模块'][0][i] in Modules:
            if (operator_info['导入模块'][1][i],operator_info['导入模块'][2][i]) not in Modules[operator_info['导入模块'][0][i]]:
                Modules[operator_info['导入模块'][0][i]].append((operator_info['导入模块'][1][i],operator_info['导入模块'][2][i]))
        else:
            Modules[operator_info['导入模块'][0][i]] = [(operator_info['导入模块'][1][i],operator_info['导入模块'][2][i])]
    # 书写导言区
    File = open(file_path,mode='a',encoding='utf-8')
    if to_truncate:
        File.truncate(0)
        File.writelines('# coding=utf-8')
    if '' in Modules:
        for iModule in Modules.pop(''):
            if iModule[1]=='':
                File.writelines('\nimport '+iModule[0])
            else:
                File.writelines('\nimport '+iModule[0]+' as '+iModule[1])
    for iSuperModule in Modules:
        for jSubModule,jModuleName in Modules[iSuperModule]:
            if jModuleName=='':
                File.writelines('\nfrom '+iSuperModule+' import '+jSubModule)
            else:
                File.writelines('\nfrom '+iSuperModule+' import '+jSubModule+' as '+jModuleName)
    # 书写函数头
    File.writelines('\ndef '+operator_info['算子名称']+'('+','.join(operator_info['算子输入'])+'):')
    # 书写函数体
    File.writelines(operator_info['算子定义'])
    File.flush()
    File.close()
    return 0
# 支持中文路径的 pandas 读取 csv 文件的函数
def readCSV2Pandas(filepath_or_buffer,detect_file_encoding=False,**other_args):
    if isinstance(filepath_or_buffer,str):# 输入的是文件路径
        with open(filepath_or_buffer,mode='rb') as File:
            filepath_or_buffer = File.read()
        if detect_file_encoding:
            import chardet
            Encoding = chardet.detect(filepath_or_buffer)
            other_args['encoding'] = Encoding["encoding"]
        filepath_or_buffer = io.BytesIO(filepath_or_buffer)
    Rslt = pd.read_csv(filepath_or_buffer,**other_args)
    return Rslt
# 获取系统偏好的文本编码格式
def guessSysTextEncoding():
    import locale
    import codecs
    return codecs.lookup(locale.getpreferredencoding()).name
# 查看文件的编码格式, 检测结果格式：{'confidence': 0.99, 'encoding': 'GB2312'}
def detectFileEncoding(file_path,big_file=False,size=None):
    if big_file:
        from chardet.universaldetector import UniversalDetector
        detector = UniversalDetector()#创建一个检测对象
        with open(file_path,mode='rb') as File:
            #分块进行测试，直到达到阈值
            if size is None:
                for line in File:
                    detector.feed(line)
                    if detector.done: break
            else:
                Batch = File.read(size=size)
                while Batch:
                    detector.feed(Batch)
                    if detector.done: break
                    Batch = File.read(size=size)
        detector.close()#关闭检测对象
        return detector.result
    else:
        import chardet
        with open(file_path,mode='rb') as File:
            if size is None:
                return chardet.detect(File.read())
            else:
                return chardet.detect(File.read(size))
# 将读入CSV文件（支持中文路径），形成DataFrame
def readCSV2StdDF(file_path,index='时间',col='字符串',encoding=None):
    Options = {"index_col":0, "header":0, "detect_file_encoding":False}
    if (index=="时间") or (col=="时间"): Options.update({"infer_datetime_format":True, "parse_dates":True})
    if encoding is not None: Options.update({"detect_file_encoding":True, "encoding":encoding})
    CSVFactor = readCSV2Pandas(file_path, **Options)
    if index=='字符串':
        CSVFactor.index = [str(iID) for iID in CSVFactor.index]
    elif index=='整数':
        CSVFactor.index = [int(float(iID)) for iID in CSVFactor.index]
    elif index=="小数":
        CSVFactor.index = [float(iID) for iID in CSVFactor.index]
    if index=='字符串':
        CSVFactor.columns = [str(iID) for iID in CSVFactor.columns]
    elif index=='整数':
        CSVFactor.columns = [int(float(iID)) for iID in CSVFactor.columns]
    elif index=="小数":
        CSVFactor.columns = [float(iID) for iID in CSVFactor.columns]
    return CSVFactor
# 将CSV中的因子数据加载入内存
def loadCSVFactorData(csv_path):
    with open(csv_path,mode='rb') as File:
        if File.readline().split(b',')[0]==b'':
            Horizon = True
        else:
            Horizon = False
    if Horizon:
        try:
            CSVFactor = readCSV2Pandas(csv_path, index_col=0, header=0, encoding="utf-8", parse_dates=True, infer_datetime_format=True)
        except:
            CSVFactor = readCSV2Pandas(csv_path, detect_file_encoding=True, index_col=0, header=0, parse_dates=True, infer_datetime_format=True)
    else:
        try:
            CSVFactor = readCSV2Pandas(csv_path, header=0, index_col=[0,1], encoding="utf-8", parse_dates=True, infer_datetime_format=True)
        except:
            CSVFactor = readCSV2Pandas(csv_path, detect_file_encoding=True, header=0, index_col=[0,1], parse_dates=True, infer_datetime_format=True)
        #Columns = list(CSVFactor.columns)
        #CSVFactor = CSVFactor.set_index(Columns[:2])[Columns[2]]
        CSVFactor = Series2DataFrame(CSVFactor.iloc[:, 0])
    try:
        if CSVFactor.index.dtype==np.dtype("O"):
            CSVDT = [dt.datetime.strptime(iDT, "%Y-%m-%D") for iDT in CSVFactor.index]
        elif CSVFactor.index.is_all_dates:
            CSVDT = [iDT.to_pydatetime() for iDT in CSVFactor.index]
        else:
            raise __QS_Error__("时间序列解析失败!")
    except:
        raise __QS_Error__("时间序列解析失败!")
    CSVID = [str(iID) for iID in CSVFactor.columns]
    CSVFactor = pd.DataFrame(CSVFactor.values, index=CSVDT, columns=CSVID)
    return CSVFactor
# 将结果集写入 CSV 文件, output: {文件名: DataFrame}
def exportOutput2CSV(output, dir_path="."):
    OutputNames = list(output.keys())
    OutputNames.sort()
    for i,iOutputName in enumerate(OutputNames):
        iOutput = output[iOutputName]
        iOutput.to_csv(dir_path+os.sep+iOutputName+".csv")
    return 0
# 读取json文件
def readJSONFile(file_path):
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as File:
            FileStr = File.read()
        if FileStr!="": return json.loads(FileStr)
    return {}
# 获取 Windows 系统的桌面路径
def getWindowsDesktopPath():
    import winreg
    Key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders")
    return winreg.QueryValueEx(Key, "Desktop")[0]
# 获取 shelve 文件的后缀名
@lru_cache(maxsize=128)
def getShelveFileSuffix():
    TestDir = tempfile.TemporaryDirectory()
    with shelve.open(TestDir.name+os.sep+"TestFile") as TestFile:
        TestFile["TestData"] = 0
    Suffix = ""
    for iFile in os.listdir(TestDir.name):
        if iFile=="TestFile": iSuffix = ""
        else: iSuffix = iFile.split(".")[-1]
        if iSuffix=="dat": return "dat"
        else: Suffix = iSuffix
    return Suffix

# 文件锁
# LOCK_EX: Exclusive Lock, 拒绝其他所有进程的读取和写入的请求
# LOCK_SH: Shared Lock(默认), 这种锁会拒绝所有进程的写入请求, 包括最初设定锁的进程. 但所有的进程都可以读取被锁定的文件.
# LOCK_NB: Nonblocking Lock, 当这个值被指定时, 如果函数不能获取指定的锁会立刻返回.
# 使用 Python 的位操作符, 或操作 |, 可以将 LOCK_NB 和 LOCK_SH 或 LOCK_EX 进行或操作.
#if os.name=="nt":# Windows 系统
    #import win32con, win32file, pywintypes
    #LOCK_EX = win32con.LOCKFILE_EXCLUSIVE_LOCK
    #LOCK_SH = 0
    #LOCK_NB = win32con.LOCKFILE_FAIL_IMMEDIATELY
    #__overlapped = pywintypes.OVERLAPPED()
    #def lockFile(file, flags):
        #hfile = win32file._get_osfhandle(file.fileno())
        #win32file.LockFileEx(hfile, flags, 0, 0xffff0000, __overlapped)
    #def unlockFile(file):
        #hfile = win32file._get_osfhandle(file.fileno())
        #win32file.UnlockFileEx(hfile, 0, 0xffff0000, __overlapped)
#elif os.name=="posix":# Unix 系统
    #from fcntl import LOCK_EX, LOCK_SH, LOCK_NB
    #def lockFile(file, flags):
        #fcntl.flock(file.fileno(), flags)
    #def unlockFile(file):
        #fcntl.flock(file.fileno(), fcntl.LOCK_UN)