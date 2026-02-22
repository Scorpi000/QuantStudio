import html
import datetime as dt
from typing import List, Any, Optional

import numpy as np
import pandas as pd

from QuantStudio.Core import __QS_Error__
from QuantStudio.Core.Node import Node, __QS_Context__
from QuantStudio.Core.FactorDB import FactorDB
from QuantStudio.Core.Factor import Factor, FactorContext, FactorLocalContext, FactorInitData
from QuantStudio.Core.CalcEngine import __QS_Engine__, Engine
from QuantStudio.Core.QSObject import Panel
from QuantStudio.Tools.IDFun import testIDFilterStr
from QuantStudio.Tools.DataTypeConversionFun import dict2html
from QuantStudio.Tools.DataTypeFun import dict2id

# 因子表, 接口类
# 因子表可看做一个独立的数据集或命名空间, 可看做 Panel(items=[因子], major_axis=[时间点], minor_axis=[ID])
# 因子表的数据有三个维度: 时间点, ID, 因子
# 时间点数据类型是 datetime.datetime, ID 和因子名称的数据类型是 str
# 不支持某个操作时, 方法产生错误
# 没有相关数据时, 方法返回 None
class FactorTable(Node):
    """因子表"""

    def __init__(self, fdb: FactorDB, args: dict={}, config_file=None, **kwargs):
        self._FactorDB = fdb # 因子表所属的因子库
        self._QS_PrepareIgnoredArgs: tuple = tuple()  # 决定 PrepareID 不需要的参数集，如果为空，表示所有参数都需要
        self._QS_LookbackArgs: tuple = ("LookBack",)
        self._QS_RawDataMaskCols: list = ["QS_ID", "QS_DT"]
        return super().__init__(args=args, config_file=config_file, **kwargs)

    @property
    def Name(self):
        return self._QSArgs.Name

    @property
    def FactorDB(self) -> FactorDB:
        return self._FactorDB

    @property
    def PrepareID(self):
        if not self._QS_PrepareIgnoredArgs: return self.QSID
        if (not getattr(self, "_QS_ID", None)) or (not getattr(self, "_QS_PrepareID", None)):
            DumpedMdl = self.model_dump()
            DumpedMdl["__qsargs__"] = {iArg: iVal for iArg, iVal in DumpedMdl["__qsargs__"].items() if iArg not in self._QS_PrepareIgnoredArgs}
            self._QS_PrepareID = dict2id(DumpedMdl)
        return self._QS_PrepareID

    # -------------------------------表的信息---------------------------------
    # 获取表的元数据
    def getMetaData(self, key=None):
        if key is None: return pd.Series()
        return None

    # -------------------------------维度信息-----------------------------------
    # 返回所有因子名
    @property
    def FactorNames(self):
        return []

    # 获取因子对象
    def getFactor(self, ifactor_name:str, args={}):
        return Factor(ft=self, args=args | {"Name": ifactor_name}, logger=self._QS_Logger)

    # 获取因子的元数据
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None: factor_names = self.FactorNames
        if key is None:
            return pd.DataFrame(index=factor_names, dtype=np.dtype("O"))
        else:
            return pd.Series([None] * len(factor_names), index=factor_names, dtype=np.dtype("O"))

    # 获取 ID 序列
    def getID(self, ifactor_name=None, idt=None):
        return []

    # 获取 ID 的 Mask, 返回: Series(True or False, index=[ID])
    def getIDMask(self, idt, ids=None, id_filter_str=None):
        if ids is None: ids = self.getID(idt=idt)
        if not id_filter_str: return pd.Series(True, index=ids)
        CompiledIDFilterStr, IDFilterFactors = testIDFilterStr(id_filter_str, self.FactorNames)
        if CompiledIDFilterStr is None: raise __QS_Error__("过滤条件字符串有误!")
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt]).loc[:, idt, :]
        return eval(CompiledIDFilterStr)

    # 获取过滤后的 ID
    def getFilteredID(self, idt, ids=None, id_filter_str=None):
        if not id_filter_str: return self.getID(idt=idt)
        if ids is None: ids = self.getID(idt=idt)
        CompiledIDFilterStr, IDFilterFactors = testIDFilterStr(id_filter_str, self.FactorNames)
        if CompiledIDFilterStr is None: raise __QS_Error__("过滤条件字符串有误!")
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt]).loc[:, idt, :]
        return eval("temp[" + CompiledIDFilterStr + "].index.tolist()")

    # 获取时间点序列
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, **kwargs):
        return []

    # 准备原始数据的接口
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        return None

    # 计算数据的接口, 返回: Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def __QS_calcData__(self, raw_data, factor_names, ids, dts):
        return None

    # 读取数据, 返回: Panel(item=[因子], major_axis=[时间点], minor_axis=[ID])
    def readData(self, factor_names, ids, dts, **kwargs):
        if not __QS_Context__:
            return self.__QS_calcData__(raw_data=self.__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts), factor_names=factor_names, ids=ids, dts=dts)
        else: Context = __QS_Context__[-1]
        if not __QS_Engine__: ExecEngine = Engine()
        else: ExecEngine = __QS_Engine__[-1]
        LocalContext = FactorLocalContext(DTs=dts, IDs=ids)
        Rslt = ExecEngine.run([self.getFactor(ifactor_name=iFactorName) for iFactorName in factor_names], Context, fwd_data_list=[LocalContext], init_data_list=[{"dt_range": (dts[0], dts[-1]), "section_ids": kwargs.get("section_ids", ids)}])
        return Panel({Rslt[i] for i, iFactorName in enumerate(factor_names)})
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.getFactor(key)
        elif isinstance(key, tuple):
            key += (slice(None),) * (3 - len(key))
        else:
            key = (key, slice(None), slice(None))
        if len(key) > 3: raise IndexError("QuantStudio.Core.FactorTable: Too many indexers")
        FactorNames, DTs, IDs = key
        if FactorNames == slice(None):
            FactorNames = self.FactorNames
        elif isinstance(FactorNames, str):
            FactorNames = [FactorNames]
        if DTs == slice(None):
            DTs = None
        elif isinstance(DTs, dt.datetime):
            DTs = [DTs]
        if IDs == slice(None):
            IDs = None
        elif isinstance(IDs, str):
            IDs = [IDs]
        Data = self.readData(FactorNames, IDs, DTs)
        return Data.loc[key]

    def _repr_html_(self):
        HTML = f"<b>名称</b>: {html.escape(self.Name)}<br/>"
        HTML += f"<b>来源因子库</b>: {html.escape(self.FactorDB.Name) if self.FactorDB is not None else ''}<br/>"
        HTML += f"<b>因子列表</b>: {html.escape(str(self.FactorNames))}<br/>"
        MetaData = self.getMetaData()
        MetaData = MetaData[~MetaData.index.str.contains("_QS")]
        HTML += f"<b>元信息</b>: {dict2html(MetaData)}"
        return HTML + super()._repr_html_()

    def __QS_saveRawData__(self, raw_data, key, target_fields, pid_ids, context: FactorContext, **kwargs):
        if raw_data is None: return 0
        Cache = context.FactorDataCache
        MaskCols = raw_data.columns.intersection(self._QS_RawDataMaskCols).tolist()
        CommonCols = raw_data.columns.difference(target_fields).tolist()
        for iFactorName in target_fields:
            iRawData = raw_data.loc[:, CommonCols + [iFactorName]]
            iKey = key + "-" + iFactorName
            iOldData = Cache.readRawData(iKey, target_fields=None, pids=None)
            if iOldData:
                iOldData = iOldData["RawData"]
                iOldData["QS_Mask"] = 1
                iRawData = pd.merge(iRawData, iOldData.loc[:, [*MaskCols, "QS_Mask"]], how="left", left_on=MaskCols, right_on=MaskCols)
                iOldData.pop("QS_Mask")
                iRawData = pd.concat([iOldData, iRawData[iRawData.pop("QS_Mask").isnull()]], ignore_index=True).sort_values(MaskCols)
            Cache.writeRawData(iKey, {"RawData": iRawData}, pid_ids, id_col="QS_ID", if_exists="replace")

    def init_compute(self, path: List[str], init_data: FactorInitData, context: FactorContext) -> List[Any]:
        PrepareData = {
            "FactorNames": [],
            "DTRange": init_data["dt_range"],
            "SectionIDs": init_data["section_ids"],
            "Args": self._QSArgs.to_dict(repr=False)
        }
        _, PrepareData = context.PrepareNodeDict.setdefault(self.PrepareID, (self.QSID, PrepareData))
        if init_data.SubFactorName not in PrepareData["FactorNames"]:
            PrepareData["FactorNames"].append(init_data.SubFactorName)
        PrepareData["DTRange"] = (min(init_data.DTRange[0], PrepareData["DTRange"][0]), max(init_data.DTRange[1], PrepareData["DTRange"][1]))
        for iArg in self._QS_LookbackArgs:
            if hasattr(self._QSArgs, iArg):
                PrepareData["Args"][iArg] = max(PrepareData["Args"][iArg], self._QSArgs[iArg])
        return super().init_compute(path=path, init_data=init_data, context=context)

    def prepare_compute(self, prepare_data: Any, context: FactorContext) -> None:
        FactorNames = sorted(prepare_data["FactorNames"])
        RawData = self.__QS_prepareRawData__(factor_names=FactorNames, ids=prepare_data["SectionIDs"], dts=prepare_data["DTRange"], args=prepare_data["Args"])
        SectionIDs = prepare_data["SectionIDs"]
        if SectionIDs==context.DefaultSectionIDs:
            PIDIDs = context.DefaultPIDIDs
        else:
            PIDIDs = context.splitID(SectionIDs)
        self.__QS_saveRawData__(raw_data=RawData, key=self.PrepareID, target_fields=FactorNames, pid_ids=PIDIDs, context=context)
        return 0
    
    def backward_compute(self, path: List[str], bwd_data_list: List[Any], context: FactorContext, local_context: Optional[FactorLocalContext]=None) -> Any:
        return None