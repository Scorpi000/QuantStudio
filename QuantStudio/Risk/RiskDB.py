# coding=utf-8
import datetime as dt
from typing import List, Optional, Any, Self

import pandas as pd
from pydantic import Field

from QuantStudio.Core import __QS_Object__, __QS_Error__


class RiskDB(__QS_Object__):
    """
    风险数据库: 由若干张风险表组成
    风险表中存储风险矩阵: Cov, DataFrame(index=[ID], columns=[ID]), ID 是证券代码
    """

    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        Name: str = Field(default="风险数据库", frozen=True)
    
    @property
    def Name(self) -> str:
        """风险库名称"""
        return self._QSArgs.Name
    
    def connect(self) -> Self:
        """连接风险数据库"""
        return self
    
    def disconnect(self) -> int:
        """断开风险数据库"""
        return 0
    
    @property
    def TableNames(self) -> List[str]:
        """风险库中风险表名称列表"""
        return []
    
    # 返回风险表对象
    def getTable(self, table_name: str, args:dict={}):
        """获取库中的风险表对象

        Args:
            table_name: 风险表名称
            args: 传递给风险表创建时初始化的参数集

        Returns:
            风险表对象
        """
        raise NotImplementedError
    
    def __getitem__(self, table_name:str):
        return self.getTable(table_name)
    
    def setTableMetaData(self, table_name:str, key:Optional[str]=None, value:Any=None, meta_data:Optional[dict]=None):
        """设置风险表的元信息, 元信息由若干个键值对组成

        Args:
            table_name: 风险表名称
            key: 元信息键
            value: 元信息值
            meta_data: 若干组键值对元信息
        """
        raise NotImplementedError
    
    def renameTable(self, old_table_name:str, new_table_name:str):
        """重命名表

        Args:
            old_table_name: 原表名
            new_table_name: 新表名
        """
        raise NotImplementedError
    
    def deleteTable(self, table_name:str):
        """删除表

        Args:
            table_name: 表名
        """
        raise NotImplementedError
    
    def deleteDateTime(self, table_name:str, dts:List[dt.datetime]):
        """删除一张表中的某些时点的风险数据

        Args:
            table_name: 表名
            dts: 待删除的时点序列
        """
        raise NotImplementedError
    
    def writeData(self, table_name:str, idt:dt.datetime, icov:pd.DataFrame, **kwargs):
        """写入风险数据

        Args:
            table_name: 风险表名称
            idt: 待写入的时点
            icov: 风险数据
        """
        raise NotImplementedError


class FactorRDB(RiskDB):
    """
    多因子风险数据库, 由若干张多因子风险表组成
    每张多因子风险表中存储着风险矩阵
    风险矩阵可以分解成 V = X*F*X' + D 的模型, 其中:
    因子风险矩阵: FactorCov(F), Panel(items=[时点], major_axis=[因子], minor_axis=[因子])
    特异性风险: SpecificRisk(D), DataFrame(index=[时点], columns=[ID])
    因子截面数据: FactorData(X), Panel(items=[因子], major_axis=[时点], minor_axis=[ID])
    因子收益率: FactorReturn, DataFrame(index=[时点], columns=[因子])
    特异性收益率: SpecificReturn, DataFrame(index=[时点], columns=[ID])
    可选存储的数据有:
    回归统计量: Statistics, {"tValue": Series(data=统计量, index=[因子]), "FValue": double, "rSquared": double, "rSquared_Adj": double}
    """

    class __QS_ArgClass__(RiskDB.__QS_ArgClass__):
        Name: str = Field(default="多因子风险数据库", frozen=True)

    def writeData(self, table_name:str, idt:dt.datetime, factor_data:Optional[pd.DataFrame]=None, factor_cov:Optional[pd.DataFrame]=None, specific_risk:Optional[pd.Series]=None, factor_ret:Optional[pd.Series]=None, specific_ret:Optional[pd.Series]=None, **kwargs):
        """写入风险数据

        Args:
            table_name: 风险表名称
            idt: 待写入的时点
            factor_data: 因子截面数据, DataFrame(index=[ID], columns=[因子]), 其中 index 是证券代码, columns 是因子列表
            factor_cov: 因子协方差矩阵, DataFrame(index=[因子], columns=[因子]), 其中 index 和 columns 都是因子列表
            specific_risk: 特异性风险: Series(index=[ID]), 其中 index 是证券代码
            factor_ret: 因子收益率, Series(index=[因子]), 其中 index 是因子列表
            specific_ret: 特异性收益率, Series(index=[ID]), 其中 index 是证券代码
        """
        raise NotImplementedError
