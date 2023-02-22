# coding=utf-8
"""常用的辅助函数"""
import os
import re
import logging
import itertools
import datetime as dt
from multiprocessing import Process, cpu_count, Queue

import numpy as np
import pandas as pd
from progressbar import ProgressBar

from QuantStudio.Tools.DataTypeConversionFun import DictKeyValueTurn_List

def getLogger(log_dir, log_level):
    Fmt = "QuantStudio - %(asctime)s - %(levelname)s : %(message)s"
    Logger = logging.getLogger()
    if isinstance(log_dir, str) and os.path.isdir(log_dir):
        LogFile = os.path.join(log_dir, "QuantStudio_"+dt.date.today().strftime("%Y%m%d")+".log")
        LogHandler = logging.FileHandler(LogFile, mode="a")
        LogHandler.setLevel(log_level)
        LogHandler.setFormatter(logging.Formatter(Fmt))
        Logger.addHandler(LogHandler)
    else:
        logging.root.setLevel(logging.NOTSET)
        logging.basicConfig(level=log_level, format=Fmt)
        Logger.setLevel(log_level)
    return Logger

# 产生一个有效的名字
def genAvailableName(header,all_names,name_num=1,check_header=True,ignore_case=False):
    if ignore_case:
        all_names = [iName.lower() for iName in all_names]
        LowerHeader = header.lower()
    else: LowerHeader = header
    if check_header and (LowerHeader not in all_names):
        AvailableNames = [header]
        CurNum = 1
    else:
        AvailableNames = []
        CurNum = 0
    i = 1
    CurName = LowerHeader+str(i)
    while (CurNum<name_num):
        if CurName not in all_names:
            AvailableNames.append(header+str(i))
            CurNum += 1
        i += 1
        CurName = LowerHeader+str(i)
    if name_num==1: return AvailableNames[0]
    else: return AvailableNames
# 指数权重,window_len:窗长；half_life:半衰期；返回:[权重]，从大到小排列
def getExpWeight(window_len,half_life,is_unitized=True):
    ExpWeight = (0.5**(1/half_life))**np.arange(window_len)
    if is_unitized:
        return list(ExpWeight/np.sum(ExpWeight))
    else:
        return list(ExpWeight)
# 将整数n近似平均的分成m份
def distributeEqual(n,m,remainder_pos="left"):
    Quotient, Remainder = int(n/m), n%m
    Rslt = np.full(shape=(m,), fill_value=Quotient, dtype=np.int64)
    if remainder_pos=='left':
        Rslt[:Remainder] += 1
    elif remainder_pos=='right':
        Rslt[-Remainder:] += 1
    else:
        StartPos = int((m-Remainder)/2)
        Rslt[StartPos:StartPos+Remainder] += 1
    return Rslt.tolist()
# 将一个list或者tuple平均分成m段
def partitionList(data,m,n_head=0,n_tail=0):
    n = len(data)
    PartitionNode = distributeEqual(n,m)
    SubData = []
    for i in range(m):
        StartInd = sum(PartitionNode[:i])
        EndInd = StartInd + PartitionNode[i]
        StartInd = max((StartInd-n_head,0))
        EndInd = min((EndInd+n_tail,n+1))
        SubData.append(data[StartInd:EndInd])
    return SubData
# 将一个list或者tuple平均分成m段, 移动抽样方式
def partitionListMovingSampling(data,m):
    n = len(data)
    SubData = []
    for i in range(m):
        SubData.append(data[i:n:m])
    return SubData
# 给定数据类型字典，产生数值和字符串因子列表
def getFactorList(data_type):
    DataTypeFactor = DictKeyValueTurn_List(data_type)
    NumFactorList = []
    StrFactorList = []
    if 'string' in DataTypeFactor:
        StrFactorList += DataTypeFactor['string']
    if 'double' in DataTypeFactor:
        NumFactorList += DataTypeFactor['double']
    if 'int' in DataTypeFactor:
        NumFactorList += DataTypeFactor['int']
    if NumFactorList==[]:
        NumFactorList = list(data_type.keys())
    if StrFactorList==[]:
        StrFactorList = NumFactorList
    return (NumFactorList,StrFactorList)
# 在给定的字符串列表str_list中寻找第一个含有name_list中给定字符串的字符串名字,如果没有找到，返回str_list的第一个元素
def searchNameInStrList(str_list,name_list):
    Rslt = None
    for iStr in str_list:
        for iNameStr in name_list:
            if iStr.find(iNameStr)!=-1:
                Rslt = iStr
                break
        if Rslt is not None:
            break
    if Rslt is None:
        Rslt = str_list[0]
    return Rslt
# 将多分类转换成单分类
# multi_class: array, 每一列是一个分类数据
# sep: 链接多个分类的字符，如果为 None，则使用"0,1,2,..."作为分类标识
def changeMultiClass2SingleClass(multi_class, sep=None):
    MultiClass = []
    for i in range(multi_class.shape[1]):
        MultiClass.append(pd.unique(multi_class[:,i]).tolist())
    SingleClassData = np.empty(shape=(multi_class.shape[0],),dtype="O")
    ClassDict = {}
    for i,iMultiClass in enumerate(itertools.product(*MultiClass)):
        iMask = np.array([True]*multi_class.shape[0])
        if sep is not None:
            iSingleClass = sep.join(map(str,iMultiClass))
        else:
            iSingleClass = str(i)
        for j,jSubClass in enumerate(iMultiClass):
            if pd.notnull(jSubClass):
                iMask = iMask & (multi_class[:,j]==jSubClass)
            else:
                iMask = iMask & pd.isnull(multi_class[:,j])
        SingleClassData[iMask] = iSingleClass
        ClassDict[iSingleClass] = iMultiClass
    return (SingleClassData,ClassDict)
# 给定某一分类subclass, 返回class_data的属于该类别的Mask, 如果subclass是None，返回全选的Mask
# subclass: [类别名称], 例如: ['银行', '大盘']
# class_data: 类别数据, DataFrame(columns=[分类名称]) 或者 array
def getClassMask(subclass,class_data):
    if isinstance(class_data, np.ndarray):
        Mask = np.array([True]*class_data.shape[0])
    else:
        Mask = pd.Series(True,index=class_data.index)
    if subclass is None:
        return Mask
    if isinstance(class_data, np.ndarray):
        for j,jSubClass in enumerate(subclass):
            if pd.notnull(jSubClass):
                Mask = Mask & (class_data[:,j]==jSubClass)
            else:
                Mask = Mask & pd.isnull(class_data[:,j])
    else:
        for j,jSubClass in enumerate(subclass):
            if pd.notnull(jSubClass):
                Mask = Mask & (class_data.iloc[:,j]==jSubClass)
            else:
                Mask = Mask & pd.isnull(class_data.iloc[:,j])
    return Mask

# 使得两个Series相匹配, 即 index 一致, 缺失的按照指定值填充
def match2Series(s1, s2, fillna=0.0):
    AllIndex = list(set(s1.index).union(set(s2.index)))
    if s1.shape[0]>0:
        s1 = s1[AllIndex]
        s1[pd.isnull(s1)] = fillna
    else: s1 = pd.Series(fillna, index=AllIndex)
    if s2.shape[0]>0:
        s2 = s2[AllIndex]
        s2[pd.isnull(s2)] = fillna
    else: s2 = pd.Series(fillna, index=AllIndex)
    return (s1, s2)
# 返回序列1在序列2中的位置
def getListIndex(s1, s2):
    Index = pd.Series(np.arange(0,len(s2)),index=s2)
    return list(Index.ix[s1].values)
# 用 join_str 连接列表, 不要求全部为字符串
def joinList(target_list, join_str):
    if not target_list:
        return ""
    Str = str(target_list[0])
    for iVal in target_list[1:]:
        Str += join_str+str(iVal)
    return Str
# 对总数 n 分配维度
def allocateDim(n, n_dim=2):
    DimAllocation = np.zeros(n_dim, dtype=int) + int(n**(1/n_dim))
    i = 0
    while np.prod(DimAllocation)<n:
        DimAllocation[i] = DimAllocation[i] + 1
        i = i+1
    return DimAllocation

# 以多进程的方式启动程序, target_fun:目标函数, 参数为(arg); main2sub_queue(sub2main_queue)可取值: None, Single, Multiple
def startMultiProcess(pid="0", n_prc=cpu_count(), target_fun=None, arg={}, partition_arg=[], n_partition_head=0, n_partition_tail=0, main2sub_queue="None", sub2main_queue="None", daemon=None):
    arg = arg.copy()
    PIDs = [pid+"-"+str(i) for i in range(n_prc)]
    if main2sub_queue=='Single': Main2SubQueue = Queue()
    elif main2sub_queue=='Multiple': Main2SubQueue = {iPID:Queue() for iPID in PIDs}
    else: Main2SubQueue = None
    if sub2main_queue=='Single': Sub2MainQueue = Queue()
    elif sub2main_queue=='Multiple': Sub2MainQueue = {iPID:Queue() for iPID in PIDs}
    else: Sub2MainQueue = None
    if (partition_arg!=[]) and (n_prc>0): ArgPartition = {iPartitionArg:partitionList(arg[iPartitionArg], n_prc, n_partition_head, n_partition_tail) for iPartitionArg in partition_arg}
    Procs = {}
    for i, iPID in enumerate(PIDs):
        iArg = arg
        iArg["PID"] = iPID
        if (partition_arg!=[]):
            for iPartitionArg in partition_arg: iArg[iPartitionArg] = ArgPartition[iPartitionArg][i]
        if sub2main_queue=='Single': iArg["Sub2MainQueue"] = Sub2MainQueue
        elif sub2main_queue=='Multiple': iArg["Sub2MainQueue"] = Sub2MainQueue[iPID]
        if main2sub_queue=='Single': iArg["Main2SubQueue"] = Main2SubQueue
        elif main2sub_queue=='Multiple': iArg["Main2SubQueue"] = Main2SubQueue[iPID]
        Procs[iPID] = Process(target=target_fun, args=(iArg,), daemon=daemon)
        Procs[iPID].start()
    return (Procs, Main2SubQueue, Sub2MainQueue)

# 以多进程的方式启动程序, target_fun:目标函数, 参数为(arg); main2sub_queue(sub2main_queue)可取值: None, Single, Multiple
def runMultiProcess(pid="0", n_prc=cpu_count(), target_fun=None, arg={}, partition_arg=[], n_partition_head=0, n_partition_tail=0, print_progress=True, daemon=None):
    Procs, Main2SubQueue, Sub2MainQueue = startMultiProcess(pid=pid, n_prc=n_prc, target_fun=target_fun, arg=arg, partition_arg=partition_arg, 
                                                            n_partition_head=n_partition_head, n_partition_tail=n_partition_tail, sub2main_queue=("Single" if print_progress else "None"), daemon=daemon)
    if print_progress:
        if partition_arg!=[]:
            TaskNums = np.fromiter(map(len, partitionList(arg[partition_arg[0]], n_prc, n_partition_head, n_partition_tail)), dtype=float)
        else:
            TaskNums = np.ones((n_prc,))
        nTask = np.sum(TaskNums)
        TaskProgs = np.zeros(TaskNums.shape)
        iProg = 0
        with ProgressBar(max_value=nTask) as ProgBar:
            while iProg<nTask:
                iPID, iSubProg, iMsg = Sub2MainQueue.get()
                if iMsg: print(iMsg)
                iIdx = int(iPID.split("-")[-1])
                TaskProgs[iIdx] = iSubProg
                iProg = np.sum(TaskProgs * TaskNums)
                ProgBar.update(min(nTask, iProg))
                if iProg>=nTask: break
    for iPID, iPrcs in Procs.items(): iPrcs.join()
    return 0

# 给定左边的标识符，查找右边与之匹配的标识符
def findRightPair(txt, left_idx, left="(", right=")"):
    LeftLen, RightLen = len(left), len(right)
    Left, EndIdx = "", 0
    for iMatch in re.finditer(r"\^|\$|\(|\[|\{|\)|\]|\}|\.|\+|\?|\*|\|", left):
        Left += left[EndIdx:iMatch.start()] + "\\" + left[iMatch.start():iMatch.end()]
        EndIdx = iMatch.end()
    Right, EndIdx = "", 0
    for iMatch in re.finditer(r"\^|\$|\(|\[|\{|\)|\]|\}|\.|\+|\?|\*|\|", right):
        Right += right[EndIdx:iMatch.start()] + "\\" + right[iMatch.start():iMatch.end()]
        EndIdx = iMatch.end()
    LeftCount, StartIdx = 1, left_idx + LeftLen
    for iMatch in re.finditer(f"{Left}|{Right}", txt[StartIdx:]):
        iMark = txt[StartIdx:][iMatch.start():iMatch.end()]
        if iMark == left:
            LeftCount += 1
        else:
            LeftCount -= 1
        if LeftCount == 0:
            return StartIdx + iMatch.start()
    return None

if __name__=="__main__":
    # print(allocateDim(13))

    Txt = "1 + (^((^(2+3)$)*4 + 5)$) / 2"
    RightIdx = findRightPair(Txt, 4, left="(^(", right=")$)")
    print(RightIdx)