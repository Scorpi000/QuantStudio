# -*- coding: utf-8 -*-
from typing import List, Any, Literal

from pydantic import Field

from QuantStudio.Core.Node import Node, Context
from QuantStudio.Core.FactorDB import FactorDB
from QuantStudio.Core.QSObject import Panel


class FactorStorer(Node):
    class __QS_ArgClass__(Node.__QS_ArgClass__):
        Name: str = Field(default="FactorStorer", frozen=True, title="名称")
        TargetFDB: FactorDB = Field(frozen=True, title="目标因子库")
        TargetTable: str = Field(frozen=True, tiltle="目标因子表")
        IfExists: Literal["update", "replace", "append"] = Field(default="update", frozen=True, title="写入方式")
    
    @property
    def FactorDB(self):
        return self._QSArgs._TargetFDB
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: Context, local_context: Any=None) -> Any:
        if hasattr(self._QSArgs.TargetFDB, "writeFactorData"):
            for i, iData in enumerate(bwd_data_list):
                iDataType = self.Deps[i].getMetaData(key="DataType")
                print(f"DEBUG {context.PID} 写入 {self.Deps[i]._QSArgs.Name}: ", iData)
                self._QSArgs.TargetFDB.writeFactorData(factor_data=iData, table_name=self._QSArgs.TargetTable, ifactor_name=self.Deps[i]._QSArgs.Name, if_exists=self._QSArgs.IfExists, data_type=iDataType)
        else:
            DataType = {iFactor._QSArgs.Name: iFactor.getMetaData(key="DataType") for iFactor in self.Deps}
            Data = Panel({self.Deps[i]._QSArgs.Name: iData for i, iData in enumerate(bwd_data_list)})
            self._QSArgs.TargetFDB.writeData(data=Data, table_name=self._QSArgs.TargetTable, if_exists=self._QSArgs.IfExists, data_type=DataType)


if __name__ == "__main__":
    pass