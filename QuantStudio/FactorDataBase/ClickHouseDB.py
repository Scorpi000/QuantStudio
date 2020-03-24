# coding=utf-8
"""基于 ClickHouse 数据库的因子库"""
import re
import os
import datetime as dt
from collections import OrderedDict

import numpy as np
import pandas as pd
from traits.api import Enum, Str, Range, Password, File, Float, Bool, ListStr, on_trait_change, Either, Date

from QuantStudio.Tools.SQLDBFun import genSQLInCondition
from QuantStudio.Tools.AuxiliaryFun import genAvailableName
from QuantStudio.Tools.DataPreprocessingFun import fillNaByLookback
from QuantStudio.Tools.QSObjects import QSSQLObject
from QuantStudio import __QS_Error__, __QS_ConfigPath__
from QuantStudio.FactorDataBase.SQLDB import SQLDB, _adjustData

def _identifyDataType(db_type, dtypes):
    if np.dtype("O") in dtypes.values: return "String"
    else: return "double"

class _WideTable(FactorTable):
    """SQLDB 宽因子表"""
    TableType = Enum("宽表", arg_type="SingleOption", label="因子表类型", order=0)
    LookBack = Float(0, arg_type="Integer", label="回溯天数", order=1)
    FilterCondition = Str("", arg_type="String", label="筛选条件", order=2)
    #DTField = Enum("dt", arg_type="SingleOption", label="时点字段", order=3)
    #IDField = Enum("code", arg_type="SingleOption", label="ID字段", order=4)
    def __init__(self, name, fdb, sys_args={}, **kwargs):
        self._DataType = fdb._TableFactorDict[name]
        self._DBDataType = fdb._TableFieldDataType[name]
        self._DBTableName = fdb.TablePrefix+fdb.InnerPrefix+name
        return super().__init__(name=name, fdb=fdb, sys_args=sys_args, **kwargs)
    def __QS_initArgs__(self):
        super().__QS_initArgs__()
        Fields = ["dt"] + self._DBDataType[self._DBDataType.str.contains("date")].index.tolist()
        self.add_trait("DTField", Enum(*Fields, arg_type="SingleOption", label="时点字段", order=4))
        StrMask = (self._DBDataType.str.contains("char") | self._DBDataType.str.contains("text"))
        Fields = ["code"] + self._DBDataType[StrMask].index.tolist()
        self.add_trait("IDField", Enum(*Fields, arg_type="SingleOption", label="ID字段", order=5))
    @property
    def FactorNames(self):
        return self._DataType.index.tolist()+["dt", "code"]
    def getFactorMetaData(self, factor_names=None, key=None):
        if factor_names is None: factor_names = self.FactorNames
        if key=="DataType": return self._DataType.append(pd.Series(["string", "string"], index=["dt","code"])).loc[factor_names]
        if key is None: return pd.DataFrame(self._DataType.append(pd.Series(["string", "string"], index=["dt","code"])).loc[factor_names], columns=["DataType"])
        else: return pd.Series([None]*len(factor_names), index=factor_names, dtype=np.dtype("O"))
    def getID(self, ifactor_name=None, idt=None, args={}):
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        SQLStr = "SELECT DISTINCT "+self._DBTableName+"."+IDField+" "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE "+self._DBTableName+"."+IDField+" IS NOT NULL "
        if idt is not None: SQLStr += "AND "+self._DBTableName+"."+DTField+"='"+idt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if ifactor_name is not None: SQLStr += "AND "+self._DBTableName+"."+ifactor_name+" IS NOT NULL "
        FilterStr = args.get("筛选条件", self.FilterCondition)
        if FilterStr: SQLStr += "AND "+FilterStr.format(Table=self._DBTableName)+" "
        SQLStr += "ORDER BY "+self._DBTableName+"."+IDField
        return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        SQLStr = "SELECT DISTINCT "+self._DBTableName+"."+DTField+" "
        SQLStr += "FROM "+self._DBTableName+" "
        SQLStr += "WHERE "+self._DBTableName+"."+DTField+" IS NOT NULL "
        if iid is not None: SQLStr += "AND "+self._DBTableName+"."+IDField+"='"+iid+"' "
        if start_dt is not None: SQLStr += "AND "+self._DBTableName+"."+DTField+">='"+start_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if end_dt is not None: SQLStr += "AND "+self._DBTableName+"."+DTField+"<='"+end_dt.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        if ifactor_name is not None: SQLStr += "AND "+self._DBTableName+"."+ifactor_name+" IS NOT NULL "
        FilterStr = args.get("筛选条件", self.FilterCondition)
        if FilterStr: SQLStr += "AND "+FilterStr.format(Table=self._DBTableName)+" "
        SQLStr += "ORDER BY "+self._DBTableName+"."+DTField
        if self._FactorDB.DBType!="sqlite3": return [iRslt[0] for iRslt in self._FactorDB.fetchall(SQLStr)]
        else: return [dt.datetime.strptime(iRslt[0], "%Y-%m-%d %H:%M:%S.%f") for iRslt in self._FactorDB.fetchall(SQLStr)]
    def __QS_genGroupInfo__(self, factors, operation_mode):
        ArgConditionGroup = {}
        ArgNames = self.ArgNames
        ArgNames.remove("回溯天数")
        ArgNames.remove("因子值类型")
        ArgNames.remove("遍历模式")
        for iFactor in factors:
            iArgConditions = (";".join([iArgName+":"+str(iFactor[iArgName]) for iArgName in ArgNames]))
            if iArgConditions not in ArgConditionGroup:
                ArgConditionGroup[iArgConditions] = {"FactorNames":[iFactor.Name], 
                                                     "RawFactorNames":{iFactor._NameInFT}, 
                                                     "StartDT":operation_mode._FactorStartDT[iFactor.Name], 
                                                     "args":iFactor.Args.copy()}
            else:
                ArgConditionGroup[iArgConditions]["FactorNames"].append(iFactor.Name)
                ArgConditionGroup[iArgConditions]["RawFactorNames"].add(iFactor._NameInFT)
                ArgConditionGroup[iArgConditions]["StartDT"] = min(operation_mode._FactorStartDT[iFactor.Name], ArgConditionGroup[iArgConditions]["StartDT"])
                ArgConditionGroup[iArgConditions]["args"]["回溯天数"] = max(ArgConditionGroup[iArgConditions]["args"]["回溯天数"], iFactor.LookBack)
        EndInd = operation_mode.DTRuler.index(operation_mode.DateTimes[-1])
        Groups = []
        for iArgConditions in ArgConditionGroup:
            StartInd = operation_mode.DTRuler.index(ArgConditionGroup[iArgConditions]["StartDT"])
            Groups.append((self, ArgConditionGroup[iArgConditions]["FactorNames"], list(ArgConditionGroup[iArgConditions]["RawFactorNames"]), operation_mode.DTRuler[StartInd:EndInd+1], ArgConditionGroup[iArgConditions]["args"]))
        return Groups
    def _genNullIDSQLStr(self, factor_names, ids, end_date, args={}):
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        DT2Str = args.get("时间转字符串", self.DT2Str)
        SubSQLStr = "SELECT "+IDField+", "
        SubSQLStr += "MAX("+DTField+") "
        SubSQLStr += "FROM "+self._DBTableName+" "
        SubSQLStr += "WHERE "+DTField+"<'"+end_date.strftime("%Y-%m-%d:%H:%M:%S.%f")+"' "
        SubSQLStr += "AND ("+genSQLInCondition(IDField, ids, is_str=True, max_num=1000)+") "
        FilterStr = args.get("筛选条件", self.FilterCondition)
        if FilterStr: SubSQLStr += "AND "+FilterStr.format(Table=self._DBTableName)+" "
        SubSQLStr += "GROUP BY "+IDField
        SQLStr = "SELECT "+DTField+", "
        SQLStr += IDField+", "
        for iField in factor_names:
            if iField=="dt": iDBDataType = "DateTime"
            elif iField=="code": iDBDataType = "String"
            else: iDBDataType = self._FactorDB._TableFieldDataType[self._Name][iField]            
            if DT2Str and (iDBDataType.lower().find("date")!=-1):
                SQLStr += "DATE_FORMAT("+iField+", '%Y-%m-%d %H:%i:%s'), "
            else:
                SQLStr += iField+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        SQLStr += "WHERE ("+IDField+", "+DTField+") IN ("+SubSQLStr+") "
        if FilterStr: SQLStr += "AND "+FilterStr.format(Table=self._DBTableName)+" "
        return SQLStr
    def __QS_prepareRawData__(self, factor_names, ids, dts, args={}):
        if (dts==[]) or (ids==[]): return pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        IDField = args.get("ID字段", self.IDField)
        DTField = args.get("时点字段", self.DTField)
        LookBack = args.get("回溯天数", self.LookBack)
        DT2Str = args.get("时间转字符串", self.DT2Str)
        if dts is not None:
            dts = sorted(dts)
            StartDate, EndDate = dts[0].date(), dts[-1].date()
            if not np.isinf(LookBack): StartDate -= dt.timedelta(LookBack)
        else:
            StartDate = EndDate = None
        # 形成 SQL 语句, 时点, ID, 因子数据
        SQLStr = "SELECT "+self._DBTableName+"."+DTField+", "
        SQLStr += self._DBTableName+"."+IDField+", "
        for iField in factor_names:
            if iField=="datetime": iDBDataType = "datetime"
            elif iField=="code": iDBDataType = "varchar(40)"
            else: iDBDataType = self._FactorDB._TableFieldDataType[self._Name][iField]
            if DT2Str and (iDBDataType.lower().find("date")!=-1):
                SQLStr += "DATE_FORMAT("+self._DBTableName+"."+iField+", '%Y-%m-%d %H:%i:%s'), "
            else:
                SQLStr += self._DBTableName+"."+iField+", "
        SQLStr = SQLStr[:-2]+" FROM "+self._DBTableName+" "
        if StartDate is not None:
            SQLStr += "WHERE "+self._DBTableName+"."+DTField+">='"+StartDate.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
            SQLStr += "AND "+self._DBTableName+"."+DTField+"<='"+EndDate.strftime("%Y-%m-%d %H:%M:%S.%f")+"' "
        else:
            SQLStr += "WHERE "+self._DBTableName+"."+DTField+" IS NOT NULL "
        if ids is not None:
            SQLStr += "AND ("+genSQLInCondition(self._DBTableName+"."+IDField, ids, is_str=True, max_num=1000)+") "
        FilterStr = args.get("筛选条件", self.FilterCondition)
        if FilterStr: SQLStr += "AND "+FilterStr.format(Table=self._DBTableName)+" "
        SQLStr += "ORDER BY "+self._DBTableName+"."+DTField+", "+self._DBTableName+"."+IDField
        if args.get("因子值类型", self.ValueType)!="scalar":
            SQLStr += ", "+self._DBTableName+"."+factor_names[0]
        RawData = self._FactorDB.fetchall(SQLStr)
        if not RawData: RawData = pd.DataFrame(columns=["QS_DT", "ID"]+factor_names)
        RawData = pd.DataFrame(np.array(RawData), columns=["QS_DT", "ID"]+factor_names)
        if (StartDate is not None) and np.isinf(LookBack):
            NullIDs = set(ids).difference(set(RawData[RawData["QS_DT"]==dt.datetime.combine(StartDate, dt.time(0))]["ID"]))
            if NullIDs:
                NullRawData = self._FactorDB.fetchall(self._genNullIDSQLStr(factor_names, list(NullIDs), StartDate, args=args))
                if NullRawData:
                    NullRawData = pd.DataFrame(np.array(NullRawData, dtype="O"), columns=["QS_DT", "ID"]+factor_names)
                    RawData = pd.concat([NullRawData, RawData], ignore_index=True)
                    RawData.sort_values(by=["QS_DT", "ID"])
        if self._FactorDB.DBType=="sqlite3": RawData["QS_DT"] = [dt.datetime.strptime(iDT, "%Y-%m-%d %H:%M:%S.%f") for iDT in RawData.pop("QS_DT")]
        return RawData
    def __QS_calcData__(self, raw_data, factor_names, ids, dts, args={}):
        if raw_data.shape[0]==0: return pd.Panel(items=factor_names, major_axis=dts, minor_axis=ids)
        raw_data = raw_data.set_index(["QS_DT", "ID"])
        DataType = self.getFactorMetaData(factor_names=factor_names, key="DataType")
        Data = {}
        for iFactorName in raw_data.columns:
            iRawData = raw_data[iFactorName].unstack()
            if DataType[iFactorName]=="double": iRawData = iRawData.astype("float")
            Data[iFactorName] = iRawData
        return _adjustData(Data, args.get("回溯天数", self.LookBack), factor_names, ids, dts)

class ClickHouseDB(SQLDB):
    """ClickHouseDB"""
    DBType = Enum("ClickHouse", arg_type="SingleOption", label="数据库类型", order=0)
    #DBName = Str("Scorpion", arg_type="String", label="数据库名", order=1)
    #IPAddr = Str("127.0.0.1", arg_type="String", label="IP地址", order=2)
    #Port = Range(low=0, high=65535, value=27017, arg_type="Integer", label="端口", order=3)
    #User = Str("root", arg_type="String", label="用户名", order=4)
    #Pwd = Password("", arg_type="String", label="密码", order=5)
    #CharSet = Enum("utf8", "gbk", "gb2312", "gb18030", "cp936", "big5", arg_type="SingleOption", label="字符集", order=6)
    Connector = Enum("default", "clickhouse-driver", arg_type="SingleOption", label="连接器", order=7)
    #IgnoreFields = ListStr(arg_type="List", label="忽略字段", order=8)
    #InnerPrefix = Str("qs_", arg_type="String", label="内部前缀", order=9)
    #CheckWriteData = Bool(False, arg_type="Bool", label="检查写入值", order=100)
    #IgnoreFields = ListStr(arg_type="List", label="忽略字段", order=101)
    #InnerPrefix = Str("qs_", arg_type="String", label="内部前缀", order=102)
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        super().__init__(sys_args=sys_args, config_file=(__QS_ConfigPath__+os.sep+"ClickHouseDBConfig.json" if config_file is None else config_file), **kwargs)
        self._TableFactorDict = {}# {表名: pd.Series(数据类型, index=[因子名])}
        self._TableFieldDataType = {}# {表名: pd.Series(数据库数据类型, index=[因子名])}
        self.Name = "ClickHouseDB"
        return
    def _connect(self):
        self._Connection = None
        if (self.Connector=="clickhouse-driver") or ((self.Connector=="default") and (self.DBType=="ClickHouse")):
            try:
                import clickhouse_driver
                if self.DSN:
                    self._Connection = clickhouse_driver.connect(dsn=self.DSN, password=self.Pwd)
                else:
                    self._Connection = clickhouse_driver.connect(user=self.User, password=self.Pwd, host=self.IPAddr, port=self.Port, database=self.DBName)
            except Exception as e:
                Msg = ("'%s' 尝试使用 clickhouse-driver 连接(%s@%s:%d)数据库 '%s' 失败: %s" % (self.Name, self.User, self.IPAddr, self.Port, self.DBName, str(e)))
                self._QS_Logger.error(Msg)
                if self.Connector!="default": raise e
            else:
                self._Connector = "clickhouse-driver"
        self._PID = os.getpid()
        return 0
    def connect(self):
        super().connect()
        nPrefix = len(self.InnerPrefix)
        SQLStr = ("SELECT table, name, type FROM system.columns WHERE database='%s' " % self.DBName)
        SQLStr += ("AND table LIKE '%s%%' " % self.InnerPrefix)
        SQLStr += "AND name NOT IN ('code', 'datetime'"
        if len(self.IgnoreFields)>0:
            SQLStr += ",'"+"','".join(self.IgnoreFields)+"') "
        else:
            SQLStr += ") "
        SQLStr += "ORDER BY table, name"
        Rslt = self.fetchall(SQLStr)
        if not Rslt:
            self._TableFieldDataType = {}
            self._TableFactorDict = {}
        else:
            self._TableFieldDataType = pd.DataFrame(np.array(Rslt), columns=["表", "因子", "DataType"]).set_index(["表", "因子"])["DataType"]
            self._TableFactorDict = self._TableFieldDataType.copy()
            Mask = (self._TableFactorDict.str.contains("String") | self._TableFactorDict.str.contains("Date"))
            self._TableFactorDict[Mask] = "string"
            self._TableFactorDict[~Mask] = "double"
            self._TableFactorDict = {iTable[nPrefix:]:self._TableFactorDict.loc[iTable] for iTable in self._TableFactorDict.index.levels[0]}
            self._TableFieldDataType = {iTable[nPrefix:]:self._TableFieldDataType.loc[iTable] for iTable in self._TableFieldDataType.index.levels[0]}
        return 0
    def renameDBTable(self, old_table_name, new_table_name):
        SQLStr = "RENAME TABLE "+self.TablePrefix+old_table_name+" TO "+self.TablePrefix+new_table_name
        try:
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 renameDBTable 将表 '%s' 重命名为 '%s' 时错误: %s" % (self.Name, old_table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 renameDBTable 将表 '%s' 重命名为 '%s'" % (self.Name, old_table_name, new_table_name))
        return 0
    def getDBTable(self):
        try:
            SQLStr = "SELECT name FROM system.tables WHERE database='"+self.DBName+"'"
            AllTables = self.fetchall(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 getDBTable 时错误: %s" % (self.Name, str(e)))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        else:
            return [rslt[0] for rslt in AllTables]
    # 创建表, field_types: {字段名: 数据类型}
    def createDBTable(self, table_name, field_types, primary_keys=[], index_fields=[]):
        SQLStr = "CREATE TABLE IF NOT EXISTS %s (" % (self.TablePrefix+table_name)
        for iField in field_types: SQLStr += "`%s` %s, " % (iField, field_types[iField])
        SQLStr += ")"
        if primary_keys:
            SQLStr += " PRIMARY KEY (`"+"`,`".join(primary_keys)+"`)"
        SQLStr += " ENGINE=MergeTree()"
        try:
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 createDBTable 在数据库中创建表 '%s' 时错误: %s" % (self.Name, table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 createDBTable 在数据库中创建表 '%s'" % (self.Name, table_name))
        return 0
    # 创建表, field_types: {字段名: 数据库数据类型}
    def createTable(self, table_name, field_types):
        FieldTypes = field_types.copy()
        FieldTypes["dt"] = field_types.pop("dt", "DATETIME NOT NULL")
        FieldTypes["code"] = field_types.pop("code", "String NOT NULL")
        self.createDBTable(self.InnerPrefix+table_name, FieldTypes, primary_keys=["dt", "code"], index_fields=["dt", "code"])
        self._TableFactorDict[table_name] = pd.Series({iFactorName: ("string" if field_types[iFactorName].find("String")!=-1 else "double") for iFactorName in field_types})
        self._TableFieldDataType[table_name] = pd.Series(field_types)
        return 0
    # 增加字段, field_types: {字段名: 数据类型}
    def addField(self, table_name, field_types):
        SQLStr = "ALTER TABLE %s " % (self.TablePrefix+table_name)
        SQLStr += "ADD COLUMN %s %s"
        try:
            for iField in field_types:
                self.execute(SQLStr % (iField, field_types[iField]))
        except Exception as e:
            Msg = ("'%s' 调用方法 addField 为表 '%s' 添加字段时错误: %s" % (self.Name, table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 addField 为表 '%s' 添加字段 ’%s'" % (self.Name, table_name, str(list(field_types.keys()))))
        return 0
    # 增加因子，field_types: {字段名: 数据库数据类型}
    def addFactor(self, table_name, field_types):
        if table_name not in self._TableFactorDict: return self.createTable(table_name, field_types)
        self.addField(self.InnerPrefix+table_name, field_types)
        NewDataType = pd.Series({iFactorName: ("string" if field_types[iFactorName].find("String")!=-1 else "double") for iFactorName in field_types})
        self._TableFactorDict[table_name] = self._TableFactorDict[table_name].append(NewDataType)
        self._TableFieldDataType[table_name] = self._TableFieldDataType[table_name].append(pd.Series(field_types))
        return 0
    # ----------------------------因子操作---------------------------------
    def renameFactor(self, table_name, old_factor_name, new_factor_name):
        if old_factor_name not in self._TableFactorDict[table_name]:
            Msg = ("因子库 '%s' 调用方法 renameFactor 错误: 因子表 '%s' 中不存在因子 '%s'!" % (self.Name, table_name, old_factor_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if (new_factor_name!=old_factor_name) and (new_factor_name in self._TableFactorDict[table_name]):
            Msg = ("因子库 '%s' 调用方法 renameFactor 错误: 新因子名 '%s' 已经存在于因子表 '%s' 中!" % (self.Name, new_factor_name, table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        self.renameField(self.InnerPrefix+table_name, old_factor_name, new_factor_name)
        self._TableFactorDict[table_name][new_factor_name] = self._TableFactorDict[table_name].pop(old_factor_name)
        self._TableFieldDataType[table_name][new_factor_name] = self._TableFieldDataType[table_name].pop(old_factor_name)
        return 0
    def deleteField(self, table_name, field_names):
        if not field_names: return 0
        try:
                SQLStr = "ALTER TABLE "+self.TablePrefix+table_name
                for iField in field_names: SQLStr += " DROP COLUMN `"+iField+"`,"
                self.execute(SQLStr[:-1])
        except Exception as e:
            Msg = ("'%s' 调用方法 deleteField 删除表 '%s' 中的字段 '%s' 时错误: %s" % (self.Name, table_name, str(field_names), str(e)))
            self._QS_Logger.error(Msg)
            raise e
        else:
            self._QS_Logger.info("'%s' 调用方法 deleteField 删除表 '%s' 中的字段 '%s'" % (self.Name, table_name, str(field_names)))
        return 0
    def deleteData(self, table_name, ids=None, dts=None, dt_ids=None):
        if table_name not in self._TableFactorDict:
            Msg = ("因子库 '%s' 调用方法 deleteData 错误: 不存在因子表 '%s'!" % (self.Name, table_name))
            self._QS_Logger.error(Msg)
            raise __QS_Error__(Msg)
        if (ids is None) and (dts is None): return self.truncateDBTable(self.InnerPrefix+table_name)
        DBTableName = self.TablePrefix+self.InnerPrefix+table_name
        SQLStr = "DELETE FROM "+DBTableName
        if dts is not None:
            DTs = [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in dts]
            SQLStr += "WHERE "+genSQLInCondition(DBTableName+".datetime", DTs, is_str=True, max_num=1000)+" "
        else:
            SQLStr += "WHERE "+DBTableName+".datetime IS NOT NULL "
        if ids is not None:
            SQLStr += "AND "+genSQLInCondition(DBTableName+".code", ids, is_str=True, max_num=1000)
        if dt_ids is not None:
            dt_ids = ["('"+iDTIDs[0].strftime("%Y-%m-%d %H:%M:%S.%f")+"', '"+iDTIDs[1]+"')" for iDTIDs in dt_ids]
            SQLStr += "AND "+genSQLInCondition("("+DBTableName+".datetime, "+DBTableName+".code)", dt_ids, is_str=False, max_num=1000)
        try:
            self.execute(SQLStr)
        except Exception as e:
            Msg = ("'%s' 调用方法 deleteData 删除表 '%s' 中数据时错误: %s" % (self.Name, table_name, str(e)))
            self._QS_Logger.error(Msg)
            raise e
        return 0
    def _adjustWriteData(self, data):
        NewData = []
        DataLen = data.applymap(lambda x: len(x) if isinstance(x, list) else 1)
        DataLenMax = DataLen.max(axis=1)
        DataLenMin = DataLen.min(axis=1)
        if (DataLenMax!=DataLenMin).sum()>0:
            self._QS_Logger.warning("'%s' 在写入因子 '%s' 时出现因子值长度不一致的情况, 将填充缺失!" % (self.Name, str(data.columns.tolist())))
        for i in range(data.shape[0]):
            iDataLen = DataLen.iloc[i]
            if iDataLen>0:
                iData = data.iloc[i].apply(lambda x: [None]*(iDataLen-len(x))+x if isinstance(x, list) else [x]*iDataLen).tolist()
                NewData.extend(zip(*iData))
        NewData = pd.DataFrame(NewData, dtype="O")
        return NewData.where(pd.notnull(NewData), None).to_records(index=False).tolist()
    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        if table_name not in self._TableFactorDict:
            FieldTypes = {iFactorName:_identifyDataType(self.DBType, data.iloc[i].dtypes) for i, iFactorName in enumerate(data.items)}
            self.createTable(table_name, field_types=FieldTypes)
            SQLStr = "INSERT INTO "+self.TablePrefix+self.InnerPrefix+table_name+" (`datetime`, `code`, "
        else:
            NewFactorNames = data.items.difference(self._TableFactorDict[table_name].index).tolist()
            if NewFactorNames:
                FieldTypes = {iFactorName:_identifyDataType(self.DBType, data.iloc[i].dtypes) for i, iFactorName in enumerate(NewFactorNames)}
                self.addFactor(table_name, FieldTypes)
            AllFactorNames = self._TableFactorDict[table_name].index.tolist()
            if self.CheckWriteData:
                OldData = self.getTable(table_name, args={"因子值类型":"list", "时间转字符串":True}).readData(factor_names=AllFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
            else:
                OldData = self.getTable(table_name, args={"时间转字符串":True}).readData(factor_names=AllFactorNames, ids=data.minor_axis.tolist(), dts=data.major_axis.tolist())
            if if_exists=="append":
                for iFactorName in AllFactorNames:
                    if iFactorName in data:
                        data[iFactorName] = OldData[iFactorName].where(pd.notnull(OldData[iFactorName]), data[iFactorName])
                    else:
                        data[iFactorName] = OldData[iFactorName]
            elif if_exists=="update":
                for iFactorName in AllFactorNames:
                    if iFactorName in data:
                        data[iFactorName] = data[iFactorName].where(pd.notnull(data[iFactorName]), OldData[iFactorName])
                    else:
                        data[iFactorName] = OldData[iFactorName]
            SQLStr = "REPLACE INTO "+self.TablePrefix+self.InnerPrefix+table_name+" (`datetime`, `code`, "
        data.major_axis = [iDT.strftime("%Y-%m-%d %H:%M:%S.%f") for iDT in data.major_axis]
        NewData = {}
        for iFactorName in data.items:
            iData = data.loc[iFactorName].stack(dropna=False)
            NewData[iFactorName] = iData
            SQLStr += "`"+iFactorName+"`, "
        NewData = pd.DataFrame(NewData).loc[:, data.items]
        Mask = pd.notnull(NewData).any(axis=1)
        NewData = NewData[Mask]
        # TODEBUG: 删除全部缺失的行
        #Mask = Mask[~Mask]
        #if Mask.shape[0]>0:
            #self.deleteData(table_name, dt_ids=Mask.index.tolist())
        if NewData.shape[0]==0: return 0
        if self._Connector in ("pyodbc", "sqlite3"):
            SQLStr = SQLStr[:-2] + ") VALUES (" + "?, " * (NewData.shape[1]+2)
        else:
            SQLStr = SQLStr[:-2] + ") VALUES (" + "%s, " * (NewData.shape[1]+2)
        SQLStr = SQLStr[:-2]+") "
        Cursor = self._Connection.cursor()
        if self.CheckWriteData:
            NewData = self._adjustWriteData(NewData.reset_index())
            Cursor.executemany(SQLStr, NewData)
        else:
            NewData = NewData.astype("O").where(pd.notnull(NewData), None)
            Cursor.executemany(SQLStr, NewData.reset_index().values.tolist())
        self._Connection.commit()
        Cursor.close()
        return 0