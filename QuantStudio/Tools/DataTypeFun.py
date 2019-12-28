# coding=utf-8
"""数据结构"""
import pickle

import numpy as np
import pandas as pd
import h5py

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
def getNestedDictValue(nested_dict, key_tuple):
    Value = nested_dict
    for iKey in key_tuple:
        if iKey not in Value:
            return None
        else:
            Value = Value[iKey]
    return Value
# 设置嵌套字典的数据
def setNestedDictValue(nested_dict, key_tuple, value):
    if not key_tuple:
        return nested_dict
    Parent = getNestedDictValue(nested_dict, key_tuple[:-1])
    Parent[key_tuple[-1]] = value
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