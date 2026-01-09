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

    # -------------------------------因子操作---------------------------------
    # 因子名, 返回: [因子名]
    @property
    def FactorNames(self):
        return []

    # 返回因子对象
    def getFactor(self, factor_name, args={}):
        raise NotImplementedError

    def __getitem__(self, factor_name):
        return self.getFactor(factor_name)

    def _repr_html_(self):
        return f"<b>名称</b>: {html.escape(self.Name)}<br/>" + super()._repr_html_()

# 支持写入的因子库, 接口类
class WritableFactorDB(FactorDB):
    """可写入的因子数据库"""
    # -------------------------------因子操作---------------------------------
    # 重命名因子
    def renameFactor(self, old_factor_name, new_factor_name, compound_factor_name=None):
        raise NotImplementedError

    # 删除因子
    def deleteFactor(self, factor_name, compound_factor_name=None):
        raise NotImplementedError

    # 设置因子的元数据. 必须具体化
    def setFactorMetaData(self, factor_name, compound_factor_name=None, key=None, value=None, meta_data=None):
        raise NotImplementedError

    # 写入数据, if_exists: append, update. data_type: dict like, {因子名:数据类型}, 必须具体化
    def writeData(self, data, compound_factor_name=None, if_exists="update", data_type={}, **kwargs):
        raise NotImplementedError

