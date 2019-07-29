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
# 将Dummy变量转化成0-1变量, dummy_var:Series(类别数据), 返回DataFrame(index=dummy_var.index,columns=所有类别)
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
# 将DataFrame转化成二重索引的Series，DataFrame的index和columns二重索引。
def DataFrame2Series(df):
    NewData = pd.DataFrame(columns=['index','column','data'])
    nIndex,nColumn = df.shape
    Columns = list(df.columns)
    for iIndex in df.index:
        iData = df.loc[iIndex]
        NewData = NewData.append(pd.DataFrame({'index':[iIndex]*nColumn,"column":Columns,"data":list(iData.values)}))
    return NewData.set_index(['index','column'])['data']
# 将二重索引的Series转化成DataFrame, 第一个索引作为index, 第二个索引作为columns
def Series2DataFrame(s,default_na=None):
    s_df = s.reset_index()
    Index = s_df.iloc[:,0].unique()
    Columns = s_df.iloc[:,1].unique()
    NewData = pd.DataFrame(index=Index,columns=Columns)
    NewData = NewData.where(pd.notnull(NewData),default_na)
    for iIndex in Index:
        iS = s.loc[iIndex]
        NewData.loc[iIndex,iS.index] = iS
    return NewData