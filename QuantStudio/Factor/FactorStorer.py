# -*- coding: utf-8 -*-
from typing import List, Any, Literal

from pydantic import Field

from QuantStudio.Core.Node import Node, Context
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Factor.FactorDB import WritableFactorDB


class FactorStorer(Node):
    class __QS_ArgClass__(Node.__QS_ArgClass__):
        Name: str = Field(default="FactorStorer", frozen=True, title="名称")
        TargetFDB: WritableFactorDB = Field(frozen=True, title="目标因子库")
        TargetTable: str = Field(frozen=True, tiltle="目标因子表")
        IfExists: Literal["update", "replace", "append"] = Field(default="update", frozen=True, title="写入方式")
        TableMeta: dict = Field(default={}, title="因子表元信息", frozen=True)
        UpdateMeta: bool = Field(default=False, frozen=True, title="更新元信息")
    
    @property
    def FactorDB(self):
        return self._QSArgs._TargetFDB
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: Context, local_context: Any=None) -> Any:
        if hasattr(self._QSArgs.TargetFDB, "writeFactorData"):
            for i, iData in enumerate(bwd_data_list):
                iDataType = self.Deps[i].getMetaData(key="DataType")
                self._QSArgs.TargetFDB.writeFactorData(factor_data=iData, table_name=self._QSArgs.TargetTable, ifactor_name=self.Deps[i]._QSArgs.Name, if_exists=self._QSArgs.IfExists, data_type=iDataType)
                self._QS_Logger.debug(f"{context.PID} 写入 {self._QSArgs.TargetFDB.Name}/{self._QSArgs.TargetTable}/{self.Deps[i]._QSArgs.Name}")
        else:
            DataType = {iFactor._QSArgs.Name: iFactor.getMetaData(key="DataType") for iFactor in self.Deps}
            Data = Panel({self.Deps[i]._QSArgs.Name: iData for i, iData in enumerate(bwd_data_list)})
            self._QSArgs.TargetFDB.writeData(data=Data, table_name=self._QSArgs.TargetTable, if_exists=self._QSArgs.IfExists, data_type=DataType)
            self._QS_Logger.debug(f"{context.PID} 写入 {self._QSArgs.TargetFDB.Name}/{self._QSArgs.TargetTable}")
        
        if self._QSArgs.UpdateMeta:
            if self._QSArgs.TableMeta: self._QSArgs.TargetFDB.setTableMetaData(self._QSArgs.TargetTable, meta_data=self._QSArgs.TableMeta)
            for iFactor in self.Deps:
                iMeta = iFactor.getMetaData(key=None)
                iMeta["SourceFactorID"] = iFactor.QSID
                self._QSArgs.TargetFDB.setFactorMetaData(self._QSArgs.TargetTable, iFactor.Name, key=None, value=None, meta_data=iMeta)


if __name__ == "__main__":
    pass