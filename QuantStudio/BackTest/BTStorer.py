# -*- coding: utf-8 -*-
"""回测结果存储器"""
from typing import List, Optional, Any

from pydantic import Field

from QuantStudio.Core.Node import Node, Context
from QuantStudio.BackTest.BackTestModel import BTNode
from QuantStudio.BackTest.BTResultDB import BTResultDB


class BTStorer(Node):
    """回测结果存储器"""

    class __QS_ArgClass__(Node.__QS_ArgClass__):
        Name: str = Field(default="BTStorer", frozen=True, title="名称")
        TargetDB: BTResultDB = Field(frozen=True, title="目标结果库")
        GroupName: Optional[str] = Field(default=None, frozen=True, title="结果组名称", description="支持路径层级, None 时按依赖节点 Name 自动生成")
        Metadata: Optional[dict] = Field(default=None, title="元信息标签", frozen=True)

    def __init__(self, deps: List[BTNode] = [], args: dict = {}, config_file: Optional[str] = None, **kwargs):
        if kwargs.get("split", True) and len(deps) > 1:
            Deps = [BTStorer(deps=[iDep], args=args, config_file=config_file, **(kwargs | {"split": False})) for iDep in deps]
            self._Splited = True
        else:
            Deps = deps
            self._Splited = False
        super().__init__(Deps, args, config_file, **kwargs)

    @property
    def ResultDB(self) -> BTResultDB:
        """目标结果库"""
        return self._QSArgs.TargetDB

    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: Context, local_context: Any = None) -> Any:
        if self._Splited:
            return
        GroupName = self._QSArgs.GroupName or self.Deps[0].Name
        self._QSArgs.TargetDB.writeResult(bwd_data_list[0], GroupName, self._QSArgs.Metadata)
        self._QS_Logger.debug(f"{context.PID} 写入 {self._QSArgs.TargetDB.Name}/{GroupName}")


def readBTResult(bt_result_db: BTResultDB, group_name: str) -> Optional[dict]:
    """从结果库读取回测结果

    Args:
        bt_result_db: 回测结果库对象
        group_name: 结果组名称

    Returns:
        嵌套 dict, None 表示不存在
    """
    return bt_result_db.readResult(group_name)


if __name__ == "__main__":
    pass
