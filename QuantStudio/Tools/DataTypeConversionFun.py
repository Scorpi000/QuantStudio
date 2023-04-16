# coding=utf-8
"""数据类型转换函数"""
import html

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
def expandListElementDataFrame(df, expand_index=True, dropna=False, empty_list_mask=False):
    ElementLen = df.iloc[:, 0].apply(lambda x: len(x)+1 if isinstance(x, list) else 0)
    EmptyListMask = (ElementLen == 1)
    Mask = (ElementLen > 1)
    data = df[Mask]
    if data.shape[0]==0:
        if dropna: df = df.dropna()
        df[:] = None
        return (df.reset_index() if expand_index else df)
    if expand_index:
        nCol = data.shape[1]
        data = data.reset_index()
        # Cols = data.columns.tolist()
        # for i in range(data.shape[1] - nCol):
        #     data[Cols[i]] = data.pop(Cols[i]).apply(lambda x: [x]) * (ElementLen[Mask].values - 1)
        # data = data.loc[:, Cols]
        data.iloc[:, :data.shape[1] - nCol] = (data.iloc[:, :data.shape[1] - nCol].applymap(lambda x: [x]).T * (ElementLen[Mask].values - 1)).T
    data = pd.DataFrame(data.sum(axis=0).tolist(), index=data.columns).T
    if dropna:
        TailRslt = df[(~Mask) & EmptyListMask]
        TailRslt[:] = None
        if expand_index:
            TailRslt = TailRslt.reset_index()
    else:
        TailRslt = df[~Mask]
        TailRslt[:] = None
        if expand_index:
            TailRslt = TailRslt.reset_index()
    # Rslt = data.append(TailRslt, ignore_index=True)
    Rslt = pd.concat((data, TailRslt), ignore_index=True)
    if not empty_list_mask: return Rslt
    RsltMask = pd.Series(False, index=data.index)
    if dropna:
        # RsltMask = RsltMask.append(EmptyListMask[(~Mask) & EmptyListMask], ignore_index=True)
        RsltMask = pd.concat((RsltMask, EmptyListMask[(~Mask) & EmptyListMask]), ignore_index=True)
    else:
        # RsltMask = RsltMask.append(EmptyListMask[~Mask], ignore_index=True)
        RsltMask = pd.concat((RsltMask, EmptyListMask[~Mask]), ignore_index=True)
    return (Rslt, RsltMask)

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

# 将 dict 类型的数据转成 html 字符串
def dict2html(dict_like, tag="ul", list_class=(list, np.ndarray), list_limit=5, dict_class=(dict, pd.Series), dict_limit=5):
    if len(dict_like)>0:
        HTML = f'<{tag} align="left">'
        for iKey, iVal in dict_like.items():
            iVal = dict_like[iKey]
            iKey = html.escape(iKey)
            if isinstance(iVal, list_class):
                iNum = len(iVal)
                if iNum>list_limit:
                    iLastNum = int(list_limit / 2)
                    iFirstNum = list_limit - iLastNum
                    iVal = list(iVal[:iFirstNum]) + ["..."] + list(iVal[-iLastNum:])
                iHTML = f'<{tag} align="left">'
                for ijVal in iVal:
                    iHTML += f"<li>{html.escape(str(ijVal))}</li>"
                iHTML += f"</{tag}>"
                HTML += f"<li>{iKey}{f'(共 {iNum} 个元素)' if iNum>list_limit else ''}: {iHTML}</li>"
            elif isinstance(iVal, dict_class):
                iKeys = list(iVal.keys())
                iNum = len(iVal)
                if iNum>dict_limit:
                    iLastNum = int(dict_limit / 2)
                    iFirstNum = dict_limit - iLastNum
                    iKeys = list(iKeys[:iFirstNum]) + ["..."] + list(iKeys[-iLastNum:])
                iHTML = f'<{tag} align="left">'
                for ijKey in iKeys:
                    ijVal = iVal.get(ijKey, "...")
                    if isinstance(ijVal, dict_class):
                        iHTML += f"<li>{html.escape(str(ijKey))}: {dict2html(ijVal, tag=tag, list_class=list_class, list_limit=list_limit, dict_class=dict_class, dict_limit=dict_limit)}</li>"
                    else:
                        iHTML += f"<li>{html.escape(str(ijKey))}: {html.escape(str(ijVal))}</li>"
                iHTML += f"</{tag}>"
                HTML += f"<li>{iKey}{f'(共 {iNum} 个元素)' if iNum>dict_limit else ''}: {iHTML}</li>"
            else:
                HTML += f"<li>{iKey}: {html.escape(str(iVal))}</li>"
        HTML += f"</{tag}>"
    else:
        HTML = "<br/>"
    return HTML

if __name__=="__main__":
    df = np.full(shape=(3, 2), fill_value=None, dtype="O")
    df[0, 0], df[0, 1] = [1, 2, 3], [2.1, 2.2, 2.3]
    df[2, 0], df[2, 1] = [4], [2.4]
    df = pd.DataFrame(df, index=pd.MultiIndex.from_tuples([("a", "a1"), ("a", "a2"), ("b", "b1")]), columns=["c1", "c2"])
    df1 = expandListElementDataFrame(df, expand_index=True)
    print("===")