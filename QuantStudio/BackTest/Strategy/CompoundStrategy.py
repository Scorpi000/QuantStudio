# -*- coding: utf-8 -*-
"""复合策略"""
import os

import pandas as pd
import numpy as np

from QuantStudio.FunLib.AuxiliaryFun import searchNameInStrList,getFactorList
from QuantStudio.FunLib.FileFun import listDirFile,writeFun2File
from QuantStudio import __QS_Error__, __QS_Object__
from QuantStudio.Strategy.StrategyTestModel import Strategy

# 自定义复合策略, TODO
class CompoundStrategy(Strategy):
    """复合策略"""
    def __init__(self, name, qs_env):
        super().__init__(name, qs_env)
        self.__QS_Type__ = "CompoundStrategy"
    def __QS_genSysArgs__(self, args=None, **kwargs):
        if args is not None:
            SysArgs._QS_MonitorChange = False
            SysArgs["多头权重配置"] = self._genWeightAllocArgs(SysArgs["多头权重配置"], SysArgs["数据源"])
            SysArgs["空头权重配置"] = self._genWeightAllocArgs(SysArgs["空头权重配置"], SysArgs["数据源"])
            if (SysArgs["多头账户"]!="无") and (SysArgs["多头账户"] not in Accounts):
                SysArgs["多头账户"] = (Accounts[0] if Accounts!=[] else "无")
            SysArgs.ArgInfo["多头账户"]["range"] = ["无"]+Accounts
            if (SysArgs["空头账户"]!="无") and (SysArgs["空头账户"] not in Accounts):
                SysArgs["空头账户"] = "无"
            SysArgs.ArgInfo["空头账户"]["range"] = ["无"]+Accounts
            SysArgs.ArgInfo["数据源"]["range"] = list(self.QSEnv.DSs.keys())
            SysArgs._QS_MonitorChange = False
            return SysArgs
        SysArgs = super().__QS_genSysArgs__(args, **kwargs)
        nSysArgs = len(SysArgs)
        DefaultDS = self.QSEnv.DSs.getDefaultDS()
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(DefaultDS.DataType)
        SysArgs._QS_MonitorChange = False
        SysArgs.update({"信号滞后期":0, 
                        "信号有效期":1,
                        "多头信号日":[],
                        "空头信号日":[],
                        "多头权重配置":self._genWeightAllocArgs(None, DefaultDS.Name), 
                        "空头权重配置":self._genWeightAllocArgs(None, DefaultDS.Name),
                        "多头账户":(Accounts[0] if Accounts!=[] else "无"),
                        "空头账户":"无",
                        "交易目标":"锁定买卖金额",
                        "数据源":DefaultDS.Name})
        SysArgs.ArgInfo.update({"信号滞后期":{"type":"Integer","order":nSysArgs,"min":0,"max":np.inf,"single_step":1},
                                "信号有效期":{"type":"Integer","order":nSysArgs+1,"min":1,"max":np.inf,"single_step":1},
                                "多头信号日":{'type':'DateList','order':nSysArgs+2},
                                "空头信号日":{'type':'DateList','order':nSysArgs+3},
                                "多头权重配置":{"type":"ArgSet","order":nSysArgs+4},
                                "空头权重配置":{"type":"ArgSet","order":nSysArgs+5},
                                "多头账户":{"type":"SingleOption","order":nSysArgs+6,"range":["无"]+Accounts},
                                "空头账户":{"type":"SingleOption","order":nSysArgs+7,"range":["无"]+Accounts},
                                "交易目标":{"type":"SingleOption","order":nSysArgs+8,"range":["锁定买卖金额","锁定目标权重","锁定目标金额"]},
                                "数据源":{"type":"SingleOption","order":nSysArgs+9,"range":list(self.QSEnv.DSs.keys()),"refresh":True,"visible":False}})
        SysArgs._QS_MonitorChange = True
        return SysArgs
    def __QS_onSysArgChanged__(self, change_type, change_info, **kwargs):
        Args, Key, Value = change_info
        if (Args is self._SysArgs) and (Key=="策略名"):
            self._SysArgs["策略名"] = Value
            self._Name = Value
            return True
        else:
            return super().__QS_onSysArgChanged__(change_type, change_info, **kwargs)
    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if (key=="_SysArgs") and (self._SysArgs["策略名"]!=self.Name):
            self._Name = self._SysArgs["策略名"]
    def __QS_start__(self):
        self.UserData = {}
        self.init()
        return 0
    def __QS_move__(self, idt, timestamp, trading_record, *args, **kwargs):
        Signal = self.genSignal(idt, timestamp, trading_record)
        self.trade(idt, timestamp, trading_record, Signal)
        return 0
    def __QS_end__(self):
        return 0
    # 可选实现
    def init(self):
        return 0
    # 可选实现
    def genSignal(self, idt, timestamp, trading_record):
        return None
    # 可选实现
    def trade(self, idt, timestamp, trading_record, signal):
        return 0
    # 可选实现
    def output(self):
        return {}