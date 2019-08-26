# coding=utf-8
import os

import numpy as np
import pandas as pd

from QuantStudio import __QS_Error__

# 将信息源文件中的表和字段信息导入信息文件
def importInfo(info_file, info_resource):
    TableInfo = pd.read_excel(info_resource, "TableInfo").set_index(["TableName"])
    FactorInfo = pd.read_excel(info_resource, 'FactorInfo').set_index(['TableName', 'FieldName'])
    try:
        from QuantStudio.Tools.DataTypeFun import writeNestedDict2HDF5
        writeNestedDict2HDF5(TableInfo, info_file, "/TableInfo")
        writeNestedDict2HDF5(FactorInfo, info_file, "/FactorInfo")
    except:
        pass
    return (TableInfo, FactorInfo)

# 更新信息文件
def updateInfo(info_file, info_resource, logger):
    if not os.path.isfile(info_file):
        logger.warning("数据库信息文件: '%s' 缺失, 尝试从 '%s' 中导入信息." % (info_file, info_resource))
    elif (os.path.getmtime(info_resource)>os.path.getmtime(info_file)):
        logger.warning("数据库信息文件: '%s' 有更新, 尝试从中导入新信息." % info_resource)
    else:
        try:
            from QuantStudio.Tools.DataTypeFun import readNestedDictFromHDF5
            return (readNestedDictFromHDF5(info_file, ref="/TableInfo"), readNestedDictFromHDF5(info_file, ref="/FactorInfo"))
        except:
            logger.warning("数据库信息文件: '%s' 损坏, 尝试从 '%s' 中导入信息." % (info_file, info_resource))
    if not os.path.isfile(info_resource): raise __QS_Error__("缺失数据库信息源文件: %s" % info_resource)
    return importInfo(info_file, info_resource)

def adjustDateTime(data, dts, fillna=False, **kwargs):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        if data.shape[0]==0:
            if isinstance(data, pd.DataFrame): data = pd.DataFrame(index=dts, columns=data.columns)
            else: data = pd.Series(index=dts)
        else:
            if fillna:
                AllDTs = data.index.union(dts)
                AllDTs = AllDTs.sort_values()
                data = data.loc[AllDTs]
                data = data.fillna(**kwargs)
            data = data.loc[dts]
    else:
        if data.shape[1]==0:
            data = pd.Panel(items=data.items, major_axis=dts, minor_axis=data.minor_axis)
        else:
            FactorNames = data.items
            if fillna:
                AllDTs = data.major_axis.union(dts)
                AllDTs = AllDTs.sort_values()
                data = data.loc[:, AllDTs, :]
                data = pd.Panel({data.items[i]:data.iloc[i].fillna(axis=0, **kwargs) for i in range(data.shape[0])})
            data = data.loc[FactorNames, dts, :]
    return data