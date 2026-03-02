import html

from pydantic import Field

from QuantStudio.Core import __QS_Object__


# 因子库, 只读, 接口类
# 数据库由若干张因子表组成
# 不支持某个操作时, 方法产生错误
# 没有相关数据时, 方法返回 None
class FactorDB(__QS_Object__):
    """因子库"""

    class __QS_ArgClass__(__QS_Object__.__QS_ArgClass__):
        Name: str = Field(default="因子库", frozen=True)

    @property
    def Name(self):
        return self._QSArgs.Name

    # ------------------------------数据源操作---------------------------------
    # 链接到数据库
    def connect(self):
        return self

    # 断开到数据库的链接
    def disconnect(self):
        return 0

    # -------------------------------表的操作---------------------------------
    # 表名, 返回: [表名]
    @property
    def TableNames(self):
        return []

    # 返回因子表对象
    def getTable(self, table_name, args={}):
        raise NotImplementedError

    def __getitem__(self, table_name):
        return self.getTable(table_name)

    def _repr_html_(self):
        return f"<b>名称</b>: {html.escape(self.Name)}<br/>" + super()._repr_html_()


# 支持写入的因子库, 接口类
class WritableFactorDB(FactorDB):
    """可写入的因子数据库"""

    # -------------------------------表的操作---------------------------------
    # 重命名表. 必须具体化
    def renameTable(self, old_table_name, new_table_name):
        raise NotImplementedError

    # 删除表. 必须具体化
    def deleteTable(self, table_name):
        raise NotImplementedError

    # 设置表的元数据. 必须具体化
    def setTableMetaData(self, table_name, key=None, value=None, meta_data=None):
        raise NotImplementedError

    # --------------------------------因子操作-----------------------------------
    # 对一张表的因子进行重命名. 必须具体化
    def renameFactor(self, table_name, old_factor_name, new_factor_name):
        raise NotImplementedError

    # 删除一张表中的某些因子. 必须具体化
    def deleteFactor(self, table_name, factor_names):
        raise NotImplementedError

    # 设置因子的元数据. 必须具体化
    def setFactorMetaData(self, table_name, ifactor_name, key=None, value=None, meta_data=None):
        raise NotImplementedError

    # 写入数据, if_exists: append, update. data_type: dict like, {因子名:数据类型}, 必须具体化
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        raise NotImplementedError