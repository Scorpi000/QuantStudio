import html
from typing import List, Optional, Any, Literal, Dict

from pydantic import Field

from QuantStudio.Core import __QS_Object__
from QuantStudio.Core.QSObject import Panel


class FactorDB(__QS_Object__):
    """因子库: 由若干张因子表组成"""

    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        Name: str = Field(default="因子库", frozen=True)

    @property
    def Name(self) -> str:
        """因子库名称"""
        return self._QSArgs.Name

    def connect(self):
        """连接到数据源"""
        return self

    # 断开到数据库的链接
    def disconnect(self) -> int:
        """跟数据源断开连接"""
        return 0

    @property
    def TableNames(self) -> List[str]:
        """因子表名称列表"""
        return []

    def getTable(self, table_name:str, args:dict={}):
        """获取库中的因子表对象

        Args:
            table_name: 因子表名称
            args: 传递给因子表创建时初始化的参数集

        Returns:
            因子表对象
        """
        raise NotImplementedError

    def __getitem__(self, table_name):
        return self.getTable(table_name)

    def _repr_html_(self):
        return f"<b>名称</b>: {html.escape(self.Name)}<br/>" + super()._repr_html_()


class WritableFactorDB(FactorDB):
    """可以写入数据的因子库"""

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

    def setTableMetaData(self, table_name:str, key:Optional[str]=None, value:Any=None, meta_data:Optional[dict]=None):
        """设置因子表的元信息, 元信息由若干个键值对组成

        Args:
            table_name: 因子表名称
            key: 元信息键
            value: 元信息值
            meta_data: 若干组键值对元信息
        """
        raise NotImplementedError

    def renameFactor(self, table_name:str, old_factor_name:str, new_factor_name:str):
        """对给定表中的因子重命名

        Args:
            table_name: 因子表名称
            old_factor_name: 原因子名
            new_factor_name: 新因子名
        """
        raise NotImplementedError

    def deleteFactor(self, table_name:str, factor_names:List[str]):
        """删除给定表中的某些因子

        Args:
            table_name: 因子表名称
            factor_names: 待删除的因子名列表
        """
        raise NotImplementedError

    def setFactorMetaData(self, table_name:str, ifactor_name:str, key:Optional[str]=None, value:Any=None, meta_data:Optional[dict]=None):
        """设置因子的元信息, 元信息由若干个键值对组成

        Args:
            table_name: 因子表名称
            ifactor_name: 因子名称
            key: 元信息键
            value: 元信息值
            meta_data: 若干组键值对元信息
        """
        raise NotImplementedError

    def writeData(self, data:Panel, table_name:str, if_exists:Literal["update", "replace", "append"]="update", data_type:Dict[str, Literal["double", "string", "object"]]={}, **kwargs):
        """写入数据

        Args:
            data: 待写入的因子数据, Panel(items=[因子], major_axis=[时点], minor_axis=[ID])
            table_name: 因子表名称
            if_exists: 如果该因子已经存在时数据写入的方式, update 表示用新数据更新原数据, append 表示不更新原数据而只增加原来没有的数据, replace 表示完全用新数据替换原数据, 等同于先删除原数据再写入
            data_type: 待写入因子的数据类型, {因子名称: "double" or "string" or "object"}, 如果 data_type 未指定某个因子的数据类型，则交由系统判定
        """
        raise NotImplementedError