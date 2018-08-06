# coding=utf-8
import numpy as np
import pandas as pd
from traits.api import Instance, Str

from QuantStudio import __QS_Object__, __QS_Error__
from QuantStudio.Tools.IDFun import testIDFilterStr

def _genStartEndDate(dts=None, start_dt=None, end_dt=None):
    if dts is not None:
        StartDate = dts[0].date()
        EndDate = dts[-1].date()
        if start_dt is not None:
            StartDate = max((StartDate, start_dt.date()))
        if end_dt is not None:
            EndDate = min((EndDate, end_dt.date()))
    else:
        StartDate = (dt.date.today() if start_dt is None else start_dt.date())
        EndDate = (dt.date.today() if end_dt is None else end_dt.date())
    return (StartDate, EndDate)
def _adjustDateTime(data, dts=None, start_dt=None, end_dt=None, fillna=False, **kwargs):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        if dts is not None:
            if fillna:
                AllDTs = data.index.union(set(dts))
                AllDTs = AllDTs.sort_values()
                data = data.ix[AllDTs]
                data = data.fillna(**kwargs)
            data = data.ix[dts]
        if start_dt is not None:
            data = data.loc[data.index>=start_dt]
        if end_dt is not None:
            data = data.loc[data.index<=end_dt]
    else:
        if dts is not None:
            if fillna:
                AllDTs = data.major_axis.union(set(dts))
                AllDTs = AllDTs.sort_values()
                data = data.ix[:, AllDTs, :]
                data = data.fillna(axis=1, **kwargs)
            data = data.ix[:, dts, :]
        if start_dt is not None:
            data = data.loc[:,data.major_axis>=start_dt]
        if end_dt is not None:
            data = data.loc[:,data.major_axis<=end_dt]
    return data

# 因子库, 只读, 接口类
# 数据库由若干张因子表组成
# 不支持某个操作时, 方法产生错误
# 没有相关数据时, 方法返回 None
class FactorDB(__QS_Object__):
    """因子库"""
    Name = Str("因子库")
    def __init__(self, sys_args={}, **kwargs):
        super().__init__(sys_args=sys_args, **kwargs)
        return
    # ------------------------------数据源操作---------------------------------
    # 链接到数据库
    def connect(self):
        return 0
    # 断开到数据库的链接
    def disconnect(self):
        return 0
    # 检查数据库是否可用
    def isAvailable(self):
        return True
    # -------------------------------表的操作---------------------------------
    # 表名, 返回: array([表名])
    @property
    def TableNames(self):
        return None
    # 返回因子表对象
    def getTable(self, table_name, args={}):
        return None
    # ------------------------------------遍历模式操作------------------------------------
    # 启动遍历模式, dts: 遍历的时间点序列, dates: 遍历的日期序列, times: 遍历的时间序列
    def start(self, dts=None, dates=None, times=None):
        return 0
    # 时间点向前移动, idt: 时间点, datetime.dateime
    def move(self, idt, *args, **kwargs):
        return 0
    # 结束遍历模式
    def end(self):
        return 0

# 支持写入的因子库, 接口类
class WritableFactorDB(FactorDB):
    """可写入的因子数据库"""
    # -------------------------------表的操作---------------------------------
    # 重命名表. 必须具体化
    def renameTable(self, old_table_name, new_table_name):
        return 0
    # 删除表. 必须具体化
    def deleteTable(self, table_name):
        return 0
    # 设置表的元数据. 必须具体化
    def setTableMetaData(self, table_name, key=None, value=None, meta_data=None):
        return 0
    # --------------------------------因子操作-----------------------------------
    # 对一张表的因子进行重命名. 必须具体化
    def renameFactor(self, table_name, old_factor_name, new_factor_name):
        return 0
    # 删除一张表中的某些因子. 必须具体化
    def deleteFactor(self, table_name, factor_names):
        return 0
    # 设置因子的元数据. 必须具体化
    def setFactorMetaData(self, table_name, ifactor_name, key=None, value=None, meta_data=None):
        return 0
    # 写入数据, if_exists: append, update, replace, skip. 必须具体化
    def writeData(self, data, table_name, if_exists='append', **kwargs):
        return 0
    # -------------------------------数据变换------------------------------------
    # 复制因子, 并不删除原来的因子
    def copyFactor(self, target_table, table_name, factor_names=None, if_exists='append', args={}):
        FT = self.getTable(table_name)
        if factor_names is None:
            factor_names = FT.FactorNames
        Data = FT.readData(factor_names=factor_names, args=args)
        return self.writeData(Data, target_table, if_exists=if_exists)
    # 时间平移, 沿着时间轴将所有数据纵向移动 lag 期, lag>0 向前移动, lag<0 向后移动, 空出来的地方填 nan
    def offsetDateTime(self, lag, table_name, factor_names=None, args={}):
        if lag==0:
            return 0
        FT = self.getTable(table_name)
        Data = FT.readData(factor_names=factor_names, args=args)
        if lag>0:
            Data.iloc[:,lag:,:] = Data.iloc[:,:-lag,:].values
            Data.iloc[:,:lag,:] = None
        elif lag<0:
            Data.iloc[:,:lag,:] = Data.iloc[:,-lag:,:].values
            Data.iloc[:,:lag,:] = None
        self.writeData(Data, table_name, if_exists='replace')
        return 0
    # 数据变换, 对原来的时间和ID序列通过某种变换函数得到新的时间序列和ID序列, 调整数据
    def changeData(self, table_name, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        if dts is None:
            return 0
        Data = self.getTable(table_name).readData(factor_names=factor_names, ids=ids, dts=dts, start_dt=start_dt, end_dt=end_dt, args=args)
        self.writeData(Data, table_name, if_exists='replace')
        return 0
    # 填充缺失值
    def fillNA(self, filled_value, table_name, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        Data = self.getTable(table_name).readData(factor_names=factor_names, ids=ids, dts=dts, start_dt=start_dt, end_dt=end_dt, args=args)
        Data.fillna(filled_value, inplace=True)
        self.writeData(Data, table_name, if_exists='update')
        return 0
    # 替换数据
    def replaceData(self, old_value, new_value, table_name, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        Data = self.getTable(table_name).readData(factor_names=factor_names, ids=ids, dts=dts, start_dt=start_dt, end_dt=end_dt, args=args)
        Data = Data.where(Data!=old_value, new_value)
        self.writeData(Data, table_name, if_exists='update')
        return 0
    # 压缩数据
    def compressData(self, table_name=None, factor_names=None):
        return 0

# 因子表, 接口类
# 因子表可看做一个独立的数据集或命名空间, 可看做 Panel(items=[因子], major_axis=[时间点], minor_axis=[ID])
# 因子表的数据有三个维度: 时间点, ID, 因子
# 时间点数据类型是 datetime.datetime, ID 和因子名称的数据类型是 str
# 不支持某个操作时, 方法产生错误
# 没有相关数据时, 方法返回 None
class FactorTable(__QS_Object__):
    Name = Str("因子表")
    FactorDB = Instance(FactorDB)
    def __init__(self, sys_args={}, **kwargs):
        super().__init__(sys_args=sys_args, **kwargs)
        return
    # -------------------------------表的操作---------------------------------
    # 判断因子是否在表中
    def __contains__(self, factor_name):
        return (factor_name in self.FactorNames)
    # 获取表的元数据
    def getMetaData(self, key=None):
        if key is None: return {}
        return None
    # --------------------------------三个维度的操作-----------------------------------
    # 返回所有因子名, array
    @property
    def FactorNames(self):
        return ()
    # 获取因子对象
    def getFactor(self, ifactor_name, args={}):
        Args = dict(self.SysArgs)
        Args.update(args)
        iFactor = Factor(ifactor_name, self.QSEnv, Args)
        iFactor.FactorTable = self
        return iFactor
    # 获取因子的元数据
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None:
            factor_names = self.FactorNames
        if key is None:
            return pd.DataFrame(index=factor_names, dtype=np.dtype("O"))
        else:
            return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    # 获取 ID 序列, 返回: array
    def getID(self, ifactor_name=None, idt=None, args={}):
        return None
    # 获取 ID 的 Mask, 返回: Series(True or False, index=[ID])
    def getIDMask(self, idt, ids=None, id_filter_str=None, args={}):
        if ids is None: ids = self.getID(idt=idt, args=args)
        if not id_filter_str: return pd.Series(True, index=ids)
        CompiledIDFilterStr, IDFilterFactors = testIDFilterStr(id_filter_str, self.FactorNames)
        if CompiledIDFilterStr is None: raise __QS_Error__("过滤条件字符串有误!")
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]
        return eval(CompiledIDFilterStr)
    # 获取过滤后的ID
    def getFilteredID(self, idt, id_filter_str=None, args={}):
        if not id_filter_str: return self.getID(idt=idt, args=args)
        CompiledIDFilterStr, IDFilterFactors = testIDFilterStr(id_filter_str, self.FactorNames)
        if CompiledIDFilterStr is None: raise __QS_Error__("过滤条件字符串有误!")
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]
        return eval("temp["+CompiledIDFilterStr+"].index.tolist()")
    # 获取时间点序列, 返回: array
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        return None
    # --------------------------------数据操作---------------------------------
    # 读取数据, 返回: Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def readData(self, factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        return None
    # 导出数据, CSV 格式
    def toCSV(self, dir_path, axis="Factor", factor_names=None, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        Data = self.readData(factor_names=factor_names, ids=ids, dts=dts, start_dt=start_dt, end_dt=end_dt, args=args)
        if axis=="Factor":
            for i, iIndex in enumerate(Data.items):
                iData = Data.iloc[i]
                iData.to_csv(dir_path+os.sep+iIndex+".csv", encoding="utf-8")
        elif axis=="DateTime":
            for i, iIndex in enumerate(Data.major_axis):
                iData = Data.iloc[:, i]
                iData.to_csv(dir_path+os.sep+iIndex.strftime("%Y%m%dT%H%M%S.%f")+".csv", encoding="utf-8")
        elif axis=="ID":
            for i, iIndex in enumerate(Data.minor_axis):
                iData = Data.iloc[:, :, i]
                iData.to_csv(dir_path+os.sep+iIndex+".csv", encoding="utf-8")
        return 0
    # ------------------------------------遍历模式操作------------------------------------
    # 启动遍历模式, dts: 遍历的时间点序列, dates: 遍历的日期序列, times: 遍历的时间序列
    def start(self, dts=None, dates=None, times=None):
        return 0
    # 时间点向前移动, idt: 时间点, datetime.dateime
    def move(self, idt, *args, **kwargs):
        return 0
    # 结束遍历模式
    def end(self):
        return 0


# 因子, 接口类
# 因子可看做一个 DataFrame(index=[时间点], columns=[ID])
# 时间点数据类型是 datetime.datetime, ID 的数据类型是 str
# 不支持某个操作时, 方法产生错误
# 没有相关数据时, 方法返回 None
class Factor(__QS_Object__):
    Name = Str("因子")
    FactorTable = Instance(FactorTable)
    def __init__(self, sys_args={}, **kwargs):
        super().__init__(sys_args=sys_args, **kwargs)
        self._NameInFT = name# 因子在所属的因子表中的名字
        return
    # 获取因子的元数据
    def getMetaData(self, key=None):
        if key is None: return {}
        return None
    # 获取 ID 序列, 返回: array
    def getID(self, idt=None, args={}):
        return None
    # 获取时间点序列, 返回: array
    def getDateTime(self, iid=None, start_dt=None, end_dt=None, args={}):
        return None
    # --------------------------------数据操作---------------------------------
    # 读取数据, 返回: Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def readData(self, ids=None, dts=None, start_dt=None, end_dt=None, args={}):
        if self.FactorTable is None: return None
        return self.FactorTable.readData(factor_names=[self._NameInFT], ids=ids, dts=dts, start_dt=start_dt, end_dt=end_dt, args=args).loc[self._NameInFT]