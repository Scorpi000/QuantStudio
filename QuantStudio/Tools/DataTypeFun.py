# coding=utf-8
"""数据结构"""
import pickle

import numpy as np
import pandas as pd
import h5py

from QuantStudio import __QS_Error__

# ---------------------嵌套字典--------------------------
# 拷贝嵌套字典, 
def copyNestedDict(nested_dict):
    Copy = {}
    for iKey in nested_dict:
        iValue = nested_dict[iKey]
        if isinstance(iValue, dict):
            Copy[iKey] = copyNestedDict(iValue)
        else:
            Copy[iKey] = iValue
    return Copy
# 从嵌套的字典中读取数据, 只要实现了 d[key] 方法即可
def getNestedDictValue(nested_dict, key_tuple, pop=False):
    if not key_tuple: return nested_dict
    Value = nested_dict
    for iKey in key_tuple[:-1]:
        if iKey not in Value:
            return None
        else:
            Value = Value[iKey]
    if pop:
        return Value.pop(key_tuple[-1], None)
    else:
        return Value.get(key_tuple[-1], None)
# 设置嵌套字典的数据
def setNestedDictValue(nested_dict, key_tuple, value):
    if not key_tuple:
        return nested_dict
    #Parent = getNestedDictValue(nested_dict, key_tuple[:-1])
    #Parent[key_tuple[-1]] = value
    iDict = nested_dict
    for iKey in key_tuple[:-1]:
        iDict = iDict.setdefault(iKey, {})
    iDict[key_tuple[-1]] = value
    return nested_dict
# 将嵌套字典转换成 [(key_tuple, value)], 只要实现了 in, d[key] 方法即可
def getNestedDictItems(nested_dict, start_key_tuple=(), non_leaf_type=dict):
    Items = []
    for iKey in nested_dict:
        iValue = nested_dict[iKey]
        if isinstance(iValue, non_leaf_type):
            Items.extend(getNestedDictItems(iValue, start_key_tuple=start_key_tuple+(iKey,)))
        else:
            Items.append((start_key_tuple+(iKey,), iValue))
    return Items
# 从嵌套字典中删除元素, 同时也删除空字典, 只要实现了 len, del d[key] 方法即可
def removeNestedDictItem(nested_dict, key_tuple):
    if not key_tuple:
        return nested_dict
    Parent = getNestedDictValue(nested_dict, key_tuple[:-1])
    if (Parent is None) or (key_tuple[-1] not in Parent):
        return nested_dict
    del Parent[key_tuple[-1]]
    key_tuple = key_tuple[:-1]
    nKey = len(key_tuple)
    for i in range(nKey):
        iParent = getNestedDictValue(nested_dict, key_tuple[:-1])
        if len(iParent[key_tuple[-1]])==0:
            del iParent[key_tuple[-1]]
        key_tuple = key_tuple[:-1]
    return nested_dict
# 将嵌套字典存入 HDF5 文件
def writeNestedDict2HDF5(nested_dict_or_value, file_path, ref):
    with h5py.File(file_path, mode="a") as File:
        if (ref in File) and (ref!="/"):
            del File[ref]
        if isinstance(nested_dict_or_value, dict):
            Group = (File.create_group(ref) if ref!="/" else File["/"])
            for iKeyTuple, iValue in getNestedDictItems(nested_dict_or_value):
                iBytes = pickle.dumps(iValue)
                Group.create_dataset("/".join(iKeyTuple), dtype=np.uint8, data=np.fromiter(iBytes, dtype=np.uint8))
        else:
            iBytes = pickle.dumps(nested_dict_or_value)
            File.create_dataset(ref, dtype=np.uint8, data=np.fromiter(iBytes, dtype=np.uint8))
# 从 HDF5 文件中读取嵌套字典
def _readNestedDictFromHDF5(h5_group_or_dataset):
    if isinstance(h5_group_or_dataset, h5py.Group):
        Data = {}
        for iKey in h5_group_or_dataset:
            Data[iKey] = _readNestedDictFromHDF5(h5_group_or_dataset[iKey])
        return Data
    else:
        iBytes = bytes(h5_group_or_dataset[...])
        return pickle.loads(iBytes)
def readNestedDictFromHDF5(file_path, ref="/"):
    with h5py.File(file_path, mode="r") as File:
        if ref not in File:
            return None
        return _readNestedDictFromHDF5(File[ref])
# 获取嵌套深度
def getNestDepth(nested_dict):
    if not isinstance(nested_dict, dict):
        return 0
    Depth = 0
    for iKey in nested_dict:
        Depth = max(Depth, getNestDepth(nested_dict[iKey])+1)
    return Depth
# 遍历嵌套字典
def traverseNestedDict(nested_dict, axis=np.inf):
    for iKey, iVal in nested_dict.items():
        if (axis<=0) or (np.isinf(axis) and (not isinstance(iVal, dict))):
            yield [iKey], iVal
        elif not isinstance(iVal, dict):
            return
        else:
            for jKeyList, ijVal in traverseNestedDict(iVal, axis=axis-1):
                yield [iKey]+jKeyList, ijVal
# 交换嵌套字典的层级
def swapaxesNestedDictDataFrame(nested_dict, axis1, axis2):
    Depth = getNestDepth(nested_dict)+2
    if (axis1>=Depth) or (axis2>=Depth):
        raise __QS_Error__("给出的交换层级: '%d'<->'%d' 超出了嵌套深度 '%d'" % (axis1, axis2, Depth))
    axis1, axis2 = min(axis1, axis2), max(axis1, axis2)
    if axis1==axis2:
        return nested_dict
    elif axis1==Depth-2:# 最后两层, DataFrame 转置操作
        for iKeyList, iVal in traverseNestedDict(nested_dict, axis=Depth-3):
            nested_dict = setNestedDictValue(nested_dict, iKeyList, iVal.T)
        return nested_dict
    NewAxis2 = axis2
    if axis2==Depth-1:
        for iKeyList, iVal in traverseNestedDict(nested_dict, axis=Depth-3):
            iVal.columns = iVal.columns.astype(str)
            nested_dict = setNestedDictValue(nested_dict, iKeyList, dict(iVal))
        NewAxis2 = Depth - 2
    elif axis2==Depth-2:
        for iKeyList, iVal in traverseNestedDict(nested_dict, axis=Depth-3):
            iVal.index = iVal.index.astype(str)
            nested_dict = setNestedDictValue(nested_dict, iKeyList, dict(iVal.T))
    NewDict = {}
    for iKeyList, iVal in traverseNestedDict(nested_dict, axis=NewAxis2):
        iKeyList[axis1], iKeyList[NewAxis2] = iKeyList[NewAxis2], iKeyList[axis1]
        NewDict = setNestedDictValue(NewDict, iKeyList, iVal)
    if axis2==Depth-1:
        for iKeyList, iVal in traverseNestedDict(NewDict, axis=Depth-3):
            NewDict = setNestedDictValue(NewDict, iKeyList, pd.DataFrame(iVal).sort_index(axis=1))
    elif axis2==Depth-2:
        for iKeyList, iVal in traverseNestedDict(NewDict, axis=Depth-3):
            NewDict = setNestedDictValue(NewDict, iKeyList, pd.DataFrame(iVal).T.sort_index(axis=0))
    return NewDict

if __name__=="__main__":
    import datetime as dt
    Bar2 = pd.DataFrame(np.random.randn(3,2), index=["中文", "b2", "b3"], columns=["中文", "我是个例子"])
    Bar2.iloc[0,0] = np.nan
    TestData = {"Bar1":{"a": {"a1": pd.DataFrame(np.random.rand(5,3)),
                                              "a2": pd.DataFrame(np.random.rand(4,3))},
                                      "b": pd.DataFrame(['a']*150,columns=['c'])},
                          "Bar2": Bar2}
    Depth = getNestDepth(TestData)
    print(Depth)
    #for iKeyList, iVal in traverseNestedDict(TestData, axis=1):
        #print(iKeyList, " : ", iVal)
    print(swapaxesNestedDictDataFrame(TestData, 1, 3))