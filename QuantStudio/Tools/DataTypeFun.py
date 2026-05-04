# coding=utf-8
"""数据结构"""
import os
import re
import json
import pickle
import hashlib
import inspect
from typing import Any, Dict, Callable, Set, Annotated, Union

import numpy as np
import pandas as pd
import h5py
from pydantic import BeforeValidator


# ---------------------特殊类型--------------------------
def validate_int_or_inf(v):
    if isinstance(v, bool):# 排除 bool（bool 是 int 子类）
        raise ValueError('bool 不被允许')
    if isinstance(v, int):
        return v
    if isinstance(v, float) and np.isinf(v):
        return np.inf
    raise ValueError(f'必须是 int 或 inf, 得到 {v}')
IntOrInf = Annotated[Union[int, float], BeforeValidator(validate_int_or_inf)]

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
def writeNestedDict2HDF5(nested_dict_or_value, file_path, ref, mode="a"):
    if not os.path.isfile(file_path):
        open(file_path, mode="a").close()# h5py 直接创建文件名包含中文的文件会报错.
    with h5py.File(file_path, mode=mode) as File:
        if (ref in File) and (ref!="/"):
            del File[ref]
        if isinstance(nested_dict_or_value, dict):
            Group = (File.create_group(ref) if ref!="/" else File["/"])
            for iKeyTuple, iValue in getNestedDictItems(nested_dict_or_value):
                iBytes = pickle.dumps(iValue)
                iDataSet = "/".join(iKeyTuple)
                if iDataSet in Group: del Group[iDataSet]
                Group.create_dataset(iDataSet, dtype=np.uint8, data=np.fromiter(iBytes, dtype=np.uint8))
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
        raise Exception("给出的交换层级: '%d'<->'%d' 超出了嵌套深度 '%d'" % (axis1, axis2, Depth))
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

# ---------------------对象唯一ID--------------------------
def serialize_function(func: Callable, visited: Set[int]) -> Dict[str, Any]:
    """
    序列化函数的核心特征：
    - 字节码、常量、变量名
    - 闭包捕获的值
    - 默认参数
    """
    code = func.__code__

    # 处理闭包
    closure = None
    if func.__closure__:
        closure = [serialize_value(cell.cell_contents, visited) for cell in func.__closure__]

    return {
        '__type__': 'function',
        # 'name': func.__name__,
        # 'doc': func.__doc__ or '',
        'code': code.co_code.hex(),# 字节码
        # 'consts': [serialize_value(c, visited) for c in code.co_consts],
        'names': code.co_names,# 引用的全局变量名
        # 'varnames': code.co_varnames,# 局部变量名
        # 'argcount': code.co_argcount,
        'defaults': serialize_value(func.__defaults__, visited),
        # 'freevars': code.co_freevars,# 闭包变量名
        'closure': closure
    }

def serialize_object(obj: Any, visited: Set[int]) -> Dict[str, Any]:
    """序列化普通对象的类和属性"""
    return {
        '__type__': 'object',
        '__class__': obj.__class__.__name__,
        '__module__': obj.__class__.__module__,
        'attributes': {k: serialize_value(v, visited) for k, v in sorted(obj.__dict__.items())}
    }

def serialize_pandas(df: pd.DataFrame | pd.Series, visited: Set[int]):
    return {
        '__type__': 'pandas.DataFrame' if isinstance(df, pd.DataFrame) else "pandas.Series",
        "value": serialize_value(df.to_dict(), visited)
    }

def serialize_numpy(a: np.ndarray, visited: Set[int]):
    return {
        "__type__": "numpy.ndarray",
        "value": serialize_value(a.flatten(order="C").tolist(), visited)
    }

def  serialize_qs_object(q: "__QS_Object__", visited: Set[int]):
    return {
        '__type__': 'qs_object',
        '__class__': q.__class__.__name__,
        '__module__': q.__class__.__module__,
        'qs_id': q.QSID
    }

def  serialize_qs_args(q: "__QS_Args__", visited: Set[int]):
    return {
        '__type__': 'qs_args',
        '__class__': q.__class__.__name__,
        '__module__': q.__class__.__module__,
        'qs_id': q.QSID
    } 

def serialize_value(value: Any, visited: Set[int] = None) -> Any:
    """
    将任意值序列化为可哈希的字典/列表/基本类型结构
    支持循环引用检测
    """
    if visited is None:
        visited = set()

    # 处理循环引用
    obj_id = id(value)
    if obj_id in visited:
        return {'__ref__': obj_id}
    visited.add(obj_id)
    
    from QuantStudio.Core import __QS_Object__, __QS_Args__
    try:
        # 基本不可变类型直接返回
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        # 容器类型递归处理
        elif isinstance(value, (list, tuple)):
            return [serialize_value(v, visited) for v in value]
        elif isinstance(value, dict):
            return {serialize_value(k, visited): serialize_value(v, visited) for k, v in sorted(value.items())}
        # 函数和方法
        elif inspect.isfunction(value):
            return serialize_function(value, visited)
        elif inspect.ismethod(value):
            return serialize_function(value.__func__, visited)
        # pandas 对象
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            return serialize_pandas(value, visited)
        # numpy 对象
        elif isinstance(value, np.ndarray):
            return serialize_numpy(value, visited)
        # __QS_Object__ 对象
        elif isinstance(value, __QS_Object__):
            return serialize_qs_object(value, visited)
        # __QS_Args__ 对象
        elif isinstance(value, __QS_Args__):
            return serialize_qs_args(value, visited)
        # 普通对象
        elif hasattr(value, '__dict__'):
            return serialize_object(value, visited)
        # 其他类型（如内置类型）
        else:
            return {
                '__type__': 'other',
                '__class__': value.__class__.__name__,
                'repr': repr(value)
            }
    finally:
        visited.remove(obj_id)

def generate_object_id(obj: Any) -> str:
    """
    为Python对象生成唯一且稳定的ID

    特性：
    1. 基于对象内容和结构生成SHA256哈希
    2. 属性相同的对象(包括函数)ID相同
    3. 跨程序重启ID保持不变
    4. 支持循环引用

    限制说明：
    1. 对于动态修改__dict__的对象，修改后ID会变化
    2. 闭包函数会捕获cell_contents的值，但不同运行时相同的闭包值会产生相同ID
    3. 外部全局变量变化不会影响函数ID（只监控函数体本身）
    4. 对于C扩展对象或内置类型，可能无法完美序列化
    5. 大对象序列化可能有性能开销，建议缓存ID

    返回：40字符的16进制哈希字符串
    """
    visited = set()
    serialized = serialize_value(obj, visited)

    # 确定性JSON序列化
    json_str = json.dumps(
        serialized,
        sort_keys=True,
        ensure_ascii=False,
        separators=(',', ':')
    )

    # 生成SHA256哈希
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

def dict2id(d):
    json_str = json.dumps(
        serialize_value(d, visited=set()),
        sort_keys=True,
        ensure_ascii=False,
        separators=(',', ':')
    )
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

# ------------ 字符串处理 ------------------
def formatPartial(text: str, values: dict) -> str:
    """
    部分格式化, 只替换 text 中 values key 指定的占位符，其他占位符保持原样。

    Args:
        text: 待格式化的字符串
        values: dict, 如 {'name': 'Alice', 'score': 100}
    
    Returns:
        格式化后的字符串
    """
    def replacer(match):
        key = match.group(1)          # 占位符里的变量名
        spec = match.group(2) or ''   # 格式说明部分（含冒号）
        if key in values:
            # 要替换的：用 format 规范处理（支持格式说明符）
            # 这里简单调用 str.format 对单个字段处理
            # 注意：需要按格式说明符格式化 values[key]
            if spec:
                # 构造临时格式字符串 ':{spec}'
                fmt_str = f'{{:{spec}}}'
                return fmt_str.format(values[key])
            else:
                return str(values[key])
        else:
            # 保留原占位符（包括格式说明）
            return f'{{{key}{spec}}}'
    
    # 正则匹配 {name} 或 {name:格式}
    pattern = r'\{([a-zA-Z_][a-zA-Z0-9_]*)(:[^\}]+)?\}'
    return re.sub(pattern, replacer, text)

if __name__ == "__main__1":
    Bar2 = pd.DataFrame(np.random.randn(3,2), index=["中文", "b2", "b3"], columns=["中文", "我是个例子"])
    Bar2.iloc[0,0] = np.nan
    TestData = {
        "Bar1": {
            "a": {
                "a1": pd.DataFrame(np.random.rand(5,3)),
                "a2": pd.DataFrame(np.random.rand(4,3))
            },
            "b": pd.DataFrame(['a']*150,columns=['c'])
        },
        "Bar2": Bar2
    }
    Depth = getNestDepth(TestData)
    print(Depth)
    for iKeyList, iVal in traverseNestedDict(TestData, axis=1):
        print(iKeyList, " : ", iVal)
    print(swapaxesNestedDictDataFrame(TestData, 1, 3))

if __name__ == "__main__":
    x1, x2 = 4, 4


    # 测试1：函数ID一致性
    def aha1():
        y1 = 3

        def func_a(a1, a2=1, **kwargs):
            global x1
            return f"hello{x1 + y1 + a2}"

        return func_a


    def aha2():
        y2 = 3

        def func_b(b1, b2=2, **kwargs):
            global x2
            return f"hello{x2 + y2 + b2}"

        return func_b


    print(serialize_function(aha1(), visited=set()))
    print(serialize_function(aha2(), visited=set()))


    def example_function(x):
        """示例函数"""
        return x * 2


    class User:
        def __init__(self, name, age, handler):
            self.name = name
            self.age = age
            self.handler = handler
            self.tags = ['user', 'active']

        def __repr__(self):
            return f"User({self.name})"


    # 测试2：对象ID一致性
    user1 = User("Alice", 30, example_function)
    user2 = User("Alice", 30, example_function)  # 相同属性
    user3 = User("Bob", 25, example_function)  # 不同属性

    id1 = generate_object_id(user1)
    id2 = generate_object_id(user2)
    id3 = generate_object_id(user3)

    print(f"相同属性对象ID一致: {id1 == id2}")  # True
    print(f"不同属性对象ID不同: {id1 == id3}")  # False

    df1 = pd.DataFrame(np.random.randn(5, 3))
    df2 = df1.copy()
    print(serialize_object(df1, visited=set()))
    print(serialize_object(df2, visited=set()))

