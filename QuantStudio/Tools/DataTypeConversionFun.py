# coding=utf-8
"""数据类型转换函数"""
import numpy as np
import pandas as pd

# 将字典的Key和Value转换,新的dict为{Value:Key}
def DictKeyValueTurn(old_dict):
    NewDict = {}
    for key in old_dict:
        NewDict[old_dict[key]]=key
    return NewDict
# 将字典的Key和Value转换,新的dict为{Value:[Key]}
def DictKeyValueTurn_List(old_dict):
    NewDict = {}
    for key in old_dict:
        if old_dict[key] in NewDict:
            NewDict[old_dict[key]].append(key)
        else:
            NewDict[old_dict[key]] = [key]
    return NewDict
# 将Dummy变量转化成0-1变量, dummy_var:Series(类别数据), 返回DataFrame(index=dummy_var.index,columns=所有类别), deprecated, 使用 pandas.get_dummies
def DummyVarTo01Var(dummy_var,ignore_na=False,ignores=[],ignore_nonstring=False):
    if dummy_var.shape[0]==0:
        return pd.DataFrame()
    NAMask = pd.isnull(dummy_var)
    if ignore_na:
        AllClasses = dummy_var[~NAMask].unique()
    else:
        dummy_var[NAMask] = np.nan
        AllClasses = dummy_var.unique()
    AllClasses = [iClass for iClass in AllClasses if (iClass not in ignores) and ((not ignore_nonstring) or isinstance(iClass,str) or pd.isnull(iClass))]
    OZVar = pd.DataFrame(0.0,index=dummy_var.index,columns=AllClasses,dtype='float')
    for iClass in AllClasses:
        if pd.notnull(iClass):
            iMask = (dummy_var==iClass)
        else:
            iMask = NAMask
        OZVar[iClass][iMask] = 1.0
    return OZVar

# 将元素为 list 的 DataFrame 扩展成元素为标量的 DataFrame, index 将被 reset
def expandListElementDataFrame(df, expand_index=True):
    ElementLen = df.iloc[:, 0].apply(lambda x: len(x)+1 if isinstance(x, list) else 0)
    Mask = (ElementLen>0)
    data = df[Mask]
    if data.shape[0]==0: return (df.reset_index() if expand_index else df)
    if expand_index:
        nCol = data.shape[1]
        data = data.reset_index()
        Cols = data.columns.tolist()
        for i in range(data.shape[1] - nCol):
            data[Cols[i]] = data.pop(Cols[i]).apply(lambda x: [x]) * (ElementLen[Mask].values - 1)
        data = data.loc[:, Cols]
    data = pd.DataFrame(data.sum(axis=0).tolist(), index=data.columns).T
    if expand_index:
        return data.append(df[~Mask].reset_index(), ignore_index=True)
    else:
        return data.append(df[~Mask], ignore_index=True)

# 全角转半角
def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code==12288:# 全角空格直接转换            
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):#全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

# 半角转全角
def strB2Q(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code==32:#半角空格直接转化                  
            inside_code = 12288
        elif inside_code>=32 and inside_code<=126:#半角字符（除空格）根据关系转化
            inside_code += 65248
        rstring += chr(inside_code)
    return rstring