# -*- coding: utf-8 -*-
"""投资组合配置型策略"""
import os

import pandas as pd
import numpy as np

from QuantStudio.FunLib.AuxiliaryFun import searchNameInStrList,getFactorList,genAvailableName,match2Series
from QuantStudio.FunLib.FileFun import listDirFile,writeFun2File
from QuantStudio.FunLib.MathFun import CartesianProduct
from .StrategyTestFun import loadCSVFilePortfolioSignal,writePortfolioSignal2CSV
from QuantStudio import QSArgs, QSError, QSObject
from .StrategyTestModel import Strategy

# 信号数据格式: pd.Series(目标权重, index=[ID]) 或者 None(表示无信号, 默认值)

# 自定义投资组合策略, 目前只支持日频
class PortfolioStrategy(Strategy):
    def __init__(self, name, qs_env):
        super().__init__(name, qs_env)
        self.AllLongSignals = {}# 存储所有生成的多头信号, {日期:信号}
        self.AllShortSignals = {}# 存储所有生成的空头信号, {日期:信号}
        return
    def _genWeightAllocArgs(self, args, ds_name):
        DefaultDS = self.QSEnv.DSs[ds_name]
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(DefaultDS.DataType)
        if args is None:
            Args = {"重配权重":False,
                    "分类因子":[],
                    "权重因子":"等权",
                    '类内权重':'等权',
                    '类别缺失':'忽略',
                    '权重缺失':'舍弃',
                    '零股处理':'清空持仓'}
            ArgInfo = {"重配权重":{'type':'Bool', 'order':0},
                       "分类因子":{'type':'MultiOption','order':1,'range':DefaultDS.FactorNames},
                       "权重因子":{"type":"SingleOption","order":2,"range":['等权']+DefaultNumFactorList},
                       "类内权重":{"type":"SingleOption","order":3,"range":['等权']+DefaultNumFactorList},
                       "类别缺失":{"type":"SingleOption","order":4,"range":["忽略","全配"]},
                       "权重缺失":{"type":"SingleOption","order":5,"range":['舍弃','填充均值']},
                       "零股处理":{"type":"SingleOption","order":6,"range":['清空持仓','保持仓位','过滤市场组合','全市场组合']}}
            return QSArgs(Args, ArgInfo, None)
        FactorNames = set(DefaultDS.FactorNames)
        NumFactorSet = set(DefaultNumFactorList)
        args._QS_MonitorChange = False
        if not set(args["分类因子"]).issubset(FactorNames):
            args["分类因子"] = []
        if args["权重因子"] not in NumFactorSet:
            args["权重因子"] = "等权"
        if args["类内权重"] not in NumFactorSet:
            args["类内权重"] = "等权"
        args.ArgInfo["分类因子"]["range"] = DefaultDS.FactorNames
        args.ArgInfo["权重因子"]["range"] = ['等权']+DefaultNumFactorList
        args.ArgInfo["类内权重"]["range"] = ['等权']+DefaultNumFactorList
        args._QS_MonitorChange = True
        return args
    def __QS_genSysArgs__(self, args=None, **kwargs):
        SysArgs = super().__QS_genSysArgs__(args, **kwargs)
        if self.QSEnv.DSs.isEmpty():
            return SysArgs
        Accounts = list(self.QSEnv.STM.Accounts)
        DefaultDS = (self.QSEnv.DSs[args["数据源"]] if (args is not None) and (args.get("数据源") in self.QSEnv.DSs) else self.QSEnv.DSs.getDefaultDS())
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(DefaultDS.DataType)
        if (args is None) or ("数据源" not in args):
            nSysArgs = len(SysArgs)
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
        args._QS_MonitorChange = False
        args["数据源"] = DefaultDS.Name
        args.ArgInfo["数据源"]["range"] = list(self.QSEnv.DSs.keys())
        args["多头权重配置"] = self._genWeightAllocArgs(args["多头权重配置"], args["数据源"])
        args["空头权重配置"] = self._genWeightAllocArgs(args["空头权重配置"], args["数据源"])
        if (args["多头账户"]!="无") and (args["多头账户"] not in Accounts):
            args["多头账户"] = (Accounts[0] if Accounts!=[] else "无")
        args.ArgInfo["多头账户"]["range"] = ["无"]+Accounts
        if (args["空头账户"]!="无") and (args["空头账户"] not in Accounts):
            args["空头账户"] = "无"
        args.ArgInfo["空头账户"]["range"] = ["无"]+Accounts
        args._QS_MonitorChange = False
        return args
    def __QS_onSysArgChanged__(self, change_type, change_info, **kwargs):
        Args, Key, Value = change_info
        if (change_type=="set") and (Key=="数据源"):# 数据源发生了变化
            Args["数据源"] = Value
            self.__QS_genSysArgs__(args=Args, **kwargs)
            return True
        return super().__QS_onSysArgChanged__(change_type, change_info, **kwargs)
    def __QS_start__(self):
        self.AllLongSignals = {}
        self.AllShortSignals = {}
        self._DS = self.QSEnv.DSs[self.SysArgs["数据源"]]
        self._LongAccount = (self.QSEnv.STM.Accounts[self.SysArgs["多头账户"]] if self.SysArgs["多头账户"]!="无" else None)
        self._ShortAccount = (self.QSEnv.STM.Accounts[self.SysArgs["空头账户"]] if self.SysArgs["空头账户"]!="无" else None)
        self._LongTradeTarget = None# 锁定的多头交易目标
        self._ShortTradeTarget = None# 锁定的空头交易目标
        self._LongSignalExcutePeriod = 0# 多头信号已经执行的期数
        self._ShortSignalExcutePeriod = 0# 空头信号已经执行的期数
        # 初始化信号滞后发生的控制变量
        self._TempData = {}
        self._TempData['StoredSignal'] = []# 暂存的信号，用于滞后发出信号
        self._TempData['LagNum'] = []# 当前日距离信号生成日的日期数
        self._LongTradeDates = (set(self._SysArgs["多头信号日"]) if self._SysArgs["多头信号日"]!=[] else None)
        self._ShortTradeDates = (set(self._SysArgs["空头信号日"]) if self._SysArgs["空头信号日"]!=[] else None)
        return super().__QS_start__()
    def __QS_move__(self, idt, timestamp, trading_record, *args, **kwargs):
        Signal = (None, None)
        self._isLongTradeDate =  ((self._LongAccount is not None) and ((self._LongTradeDates is None) or (idt in self._LongTradeDates)))
        self._isShortTradeDate = ((self._ShortAccount is not None) and ((self._ShortTradeDates is None) or (idt in self._ShortTradeDates)))
        if (self.QSEnv.STM.isDateEndTime) and (self._isLongTradeDate or self._isShortTradeDate):
            Signal = self.genSignal(idt, timestamp, trading_record)
        self.trade(idt, timestamp, trading_record, Signal)
        return 0
    def __QS_end__(self):
        self._DS, self._LongAccount, self._ShortAccount, self._LongTradeTarget, self._ShortTradeTarget = None, None, None, None, None
        self._LongTradeDates, self._ShortTradeDates = None, None
        self._TempData = {}
        return 0
    def genLongSignal(self, idt, timestamp, trading_record):
        return None
    def genShortSignal(self, idt, timestamp, trading_record):
        return None
    def genSignal(self, idt, timestamp, trading_record):
        if self._isLongTradeDate:
            LongSignal = self.genLongSignal(idt, timestamp, trading_record)
        else:
            LongSignal = None
        if self._isShortTradeDate:
            ShortSignal = self.genShortSignal(idt, timestamp, trading_record)
        else:
            ShortSignal = None
        return (LongSignal, ShortSignal)
    def trade(self, idt, timestamp, trading_record, signal):
        LongSignal, ShortSignal = signal
        if LongSignal is not None:
            if self.SysArgs['多头权重配置']['重配权重']:
                LongSignal = self._allocateWeight(idt, list(LongSignal.index), None, self.SysArgs['多头权重配置'])
            self.AllLongSignals[idt] = LongSignal
        if ShortSignal is not None:
            if self.SysArgs['空头权重配置']['重配权重']:
                ShortSignal = self._allocateWeight(idt, list(ShortSignal.index), None, self.SysArgs['空头权重配置'])
            self.AllShortSignals[idt] = ShortSignal
        LongSignal, ShortSignal = self._bufferSignal(LongSignal, ShortSignal)
        self._processLongSignal(idt, timestamp, trading_record, LongSignal)
        self._processShortSignal(idt, timestamp, trading_record, ShortSignal)
        return 0
    def _processLongSignal(self, idt, timestamp, trading_record, long_signal):
        if self._LongAccount is None:
            return 0
        LongAccount = self._LongAccount
        AccountValue = LongAccount.AccountValue
        pHolding = LongAccount.Portfolio
        pHolding = pHolding[pd.notnull(pHolding)]
        if long_signal is not None:# 有新的信号, 形成新的交易目标
            if self.SysArgs["交易目标"]=="锁定买卖金额":
                pSignalHolding, long_signal = match2Series(pHolding, long_signal, fillna=0.0)
                self._LongTradeTarget = (long_signal - pSignalHolding)*AccountValue
            elif self.SysArgs["交易目标"]=="锁定目标权重":
                self._LongTradeTarget = long_signal
            elif self.SysArgs["交易目标"]=="锁定目标金额":
                self._LongTradeTarget = long_signal*AccountValue
            self._LongSignalExcutePeriod = 0
        elif self._LongTradeTarget is not None:# 没有新的信号, 根据交易记录调整交易目标
            self._LongSignalExcutePeriod += 1
            if self._LongSignalExcutePeriod>=self.SysArgs["信号有效期"]:
                self._LongTradeTarget = None
                self._LongSignalExcutePeriod = 0
            else:
                iTradingRecord = trading_record[self.SysArgs["多头账户"]]
                iTradingRecord = iTradingRecord.set_index(["ID"]).ix[self._LongTradeTarget.index]
                if self.SysArgs["交易目标"]=="锁定买卖金额":
                    TargetChanged = iTradingRecord["数量"]*iTradingRecord["价格"]
                    TargetChanged[pd.isnull(TargetChanged)] = 0.0
                    self._LongTradeTarget = self._LongTradeTarget - TargetChanged
        # 根据交易目标下订单
        if self._LongTradeTarget is not None:
            if self.SysArgs["交易目标"]=="锁定买卖金额":
                Orders = self._LongTradeTarget
            elif self.SysArgs["交易目标"]=="锁定目标权重":
                pHolding, LongTradeTarget = match2Series(pHolding, self._LongTradeTarget, fillna=0.0)
                Orders = (LongTradeTarget - pHolding)*AccountValue
            elif self.SysArgs["交易目标"]=="锁定目标金额":
                pHolding, LongTradeTarget = match2Series(pHolding, self._LongTradeTarget, fillna=0.0)
                Orders = LongTradeTarget - pHolding*AccountValue
            Orders = Orders[pd.notnull(Orders) & (Orders!=0)]
            Orders = pd.DataFrame(Orders)
            Orders.columns = ["金额"]
            Orders["目标价"] = np.nan
            LongAccount.orderAmount(combined_order=Orders)
        return 0
    def _processShortSignal(self, idt, timestamp, trading_record, short_signal):
        if self.SysArgs["多头账户"]=="无":
            return 0
        pass
    # 为股票池配权重
    def _allocateWeight(self, idt, ids, original_ids=None, args={}):
        if original_ids is None:
            original_ids = self._DS.getID(idt,is_filtered=True)
        nID = len(ids)
        if nID==0:
            if args['零股处理']=='保持仓位':
                return None
            elif args['零股处理']=='清空持仓':
                return pd.Series([])
            elif args['零股处理']=='过滤市场组合':
                ids = original_ids
            elif args['零股处理']=='全市场组合':
                ids = self._DS.getID(idt)
        if args['分类因子']==[]:
            if args['权重因子']=='等权':
                NewSignal = pd.Series(1/nID,index=ids)
            else:
                WeightData = self._DS.getFactorData(ifactor_name=args['权重因子'],dates=[idt],ids=ids).loc[idt]
                if args['权重缺失']=='舍弃':
                    WeightData = WeightData[pd.notnull(WeightData)]
                else:
                    WeightData[pd.notnull(WeightData)] = WeightData.mean()
                WeightData = WeightData/WeightData.sum()
                NewSignal = WeightData
        else:
            ClassData = self._DS.getDateTimeData(factor_names=args['分类因子'],idt=idt,ids=original_ids)
            ClassData[pd.isnull(ClassData)] = np.nan
            AllClasses = [list(ClassData[iClass].unique()) for iClass in args['分类因子']]
            AllClasses = CartesianProduct(AllClasses)
            nClass = len(AllClasses)
            if args['权重因子']=='等权':
                ClassWeight = pd.Series([1/nClass]*nClass,dtype='float')
            else:
                ClassWeight = pd.Series(index=[i for i in range(nClass)],dtype='float')
                WeightData = self._DS.getFactorData(ifactor_name=args['权重因子'],dates=[idt],ids=original_ids).loc[idt]
                for i,iClass in enumerate(AllClasses):
                    if pd.notnull(iClass[0]):
                        iMask = (ClassData[args['分类因子'][0]]==iClass[0])
                    else:
                        iMask = pd.isnull(ClassData[args['分类因子'][0]])
                    for j,jSubClass in enumerate(iClass[1:]):
                        if pd.notnull(jSubClass):
                            iMask = iMask & (ClassData[args['分类因子'][j+1]]==jSubClass)
                        else:
                            iMask = iMask & pd.isnull(ClassData[args['分类因子'][j+1]])
                    ClassWeight.iloc[i] = WeightData[iMask].sum()
                ClassWeight[pd.isnull(ClassWeight)] = 0
                ClassTotalWeight = ClassWeight.sum()
                if ClassTotalWeight!=0:
                    ClassWeight = ClassWeight/ClassTotalWeight
            if args['类内权重']=='等权':
                WeightData = pd.Series([1],index=original_ids)
            else:
                WeightData = self._DS.getFactorData(ifactor_name=args['类内权重'],dates=[idt],ids=original_ids).loc[idt]
            SelectedClassData = ClassData.loc[ids]
            NewSignal = pd.Series()
            for i,iClass in enumerate(AllClasses):
                if pd.notnull(iClass[0]):
                    iMask = (SelectedClassData[args['分类因子'][0]]==iClass[0])
                else:
                    iMask = pd.isnull(SelectedClassData[args['分类因子'][0]])
                for j,jSubClass in enumerate(iClass[1:]):
                    if pd.notnull(jSubClass):
                        iMask = iMask & (SelectedClassData[args['分类因子'][j+1]]==jSubClass)
                    else:
                        iMask = iMask & pd.isnull(SelectedClassData[args['分类因子'][j+1]])
                iIDs = list(SelectedClassData[iMask].index)
                if (iIDs==[]) and (args['类别缺失']=='全配'):
                    if pd.notnull(iClass[0]):
                        iMask = (ClassData[args['分类因子'][0]]==iClass[0])
                    else:
                        iMask = pd.isnull(ClassData[args['分类因子'][0]])
                    for k,kSubClass in enumerate(iClass[1:]):
                        if pd.notnull(kSubClass):
                            iMask = iMask & (ClassData[args['分类因子'][k+1]]==kSubClass)
                        else:
                            iMask = iMask & pd.isnull(ClassData[args['分类因子'][k+1]])
                    iIDs = list(ClassData[iMask].index)
                elif (iIDs==[]) and (args['类别缺失']=='忽略'):
                    continue
                iSignal = WeightData.loc[iIDs]
                iSignalWeight = iSignal.sum()
                if iSignalWeight!=0:
                    iSignal = iSignal/iSignal.sum()*ClassWeight.iloc[i]
                else:
                    iSignal = iSignal*0.0
                if args['权重缺失']=='填充均值':
                    iSignal[pd.isnull(iSignal)] = iSignal.mean()
                NewSignal = NewSignal.append(iSignal[pd.notnull(iSignal) & (iSignal!=0)])
            NewSignal = NewSignal/NewSignal.sum()
        return NewSignal
    # 将信号缓存，并弹出滞后期到期的信号
    def _bufferSignal(self, long_signal, short_signal):
        if (long_signal is not None) or (short_signal is not None):
            self._TempData['StoredSignal'].append((long_signal, short_signal))
            self._TempData['LagNum'].append(-1)
        for i,iLagNum in enumerate(self._TempData['LagNum']):
            self._TempData['LagNum'][i] = iLagNum+1
        LongSignal, ShortSignal = None, None
        while self._TempData['StoredSignal']!=[]:
            if self._TempData['LagNum'][0]>=self.SysArgs['信号滞后期']:
                LongSignal, ShortSignal = self._TempData['StoredSignal'].pop(0)
                self._TempData['LagNum'].pop(0)
            else:
                break
        return (LongSignal, ShortSignal)
    # 将信号写入CSV文件
    def writeSignal2CSV(self, csv_path):
        if csv_path!='':
            writePortfolioSignal2CSV(self.AllLongSignals, csv_path)
            DirPath, CSVFileName = os.path.split(csv_path)
            TempPos = CSVFileName.find('.csv')
            if TempPos==-1:
                CSVFileName += '_Short'
            else:
                CSVFileName = CSVFileName[:TempPos]+'_Short'+CSVFileName[TempPos:]
            writePortfolioSignal2CSV(self.AllShortSignals,os.path.join(DirPath,CSVFileName))
        return 0

# CSV文件投资组合策略
class CSVFilePortfolioStrategy(PortfolioStrategy):
    def __QS_genSysArgs__(self, args=None, **kwargs):
        SysArgs = super().__QS_genSysArgs__(args, **kwargs)
        if self.QSEnv.DSs.isEmpty():
            return SysArgs
        if (args is None) or ("数据源" not in args):
            nSysArgs = len(SysArgs)
            SysArgs._QS_MonitorChange = False
            SysArgs.update({"多头信号文件":"", "空头信号文件":""})
            SysArgs.ArgInfo.update({"多头信号文件":{"type":"Path","order":nSysArgs,"operation":"Open","filter":"Excel (*.csv)"},
                                    "空头信号文件":{"type":"Path","order":nSysArgs+1,"operation":"Open","filter":"Excel (*.csv)"}})
            SysArgs._QS_MonitorChange = True
            return SysArgs
        return super().__QS_genSysArgs__(args, **kwargs)
    def __QS_start__(self):
        Rslt = super().__QS_start__()
        # 加载信号文件
        with self.QSEnv.CacheLock:
            self._TempData['FileLongSignals'] = loadCSVFilePortfolioSignal(self.SysArgs['多头信号文件'])
            self._TempData['FileShortSignals'] = loadCSVFilePortfolioSignal(self.SysArgs['空头信号文件'])
        return Rslt
    def genLongSignal(self, idt, timestamp, trading_record):
        return self.TempData['FileLongSignals'].get(idt)
    def genLongSignal(self, idt, timestamp, trading_record):
        return self.TempData['FileShortSignals'].get(idt)

# 分层筛选投资组合策略
class HierarchicalFiltrationStrategy(PortfolioStrategy):
    # 产生换手缓冲参数
    def _genTurnoverBufferArgs(self, args=None):
        if args is not None:
            return args
        Args = {"是否缓冲":False,
                "筛选上限缓冲区":0.1,
                "筛选下限缓冲区":0.1,
                "筛选数目缓冲区":0}
        ArgInfo = {"是否缓冲":{"type":"Bool","order":0},
                   "筛选上限缓冲区":{"type":"Double","order":1,"min":0,"max":1,"single_step":0.0001},
                   "筛选下限缓冲区":{"type":"Double","order":1,"min":0,"max":1,"single_step":0.0001},
                   "筛选数目缓冲区":{"type":"Integer","order":1,"min":0,"max":np.inf,"single_step":1}}
        return QSArgs(Args, ArgInfo, None)
    # 产生因子合并参数
    def _genFactorMergeArgs(self, args, ds_name):
        DS = self.QSEnv.DSs[ds_name]
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(DS.DataType)
        if args is None:
            Args = {"描述子":pd.DataFrame([("降序",1.0)],index=[DefaultNumFactorList[0]],columns=["排序方向", "合成权重"]),
                    "合成方式":"直接合成",
                    "缺失处理":"填充None"}
            ArgInfo = {"描述子":{"type":"ArgFrame", "order":0, "row":DefaultNumFactorList, "row_removable":True,
                                 "colarg_info":{"排序方向":{"type":"SingleOption","range":["降序","升序"]},"合成权重":{"type":"Double","min":0,"max":np.inf,"single_step":0.0001}},
                                 "unchecked_value":{"排序方向":"降序","合成权重":1/len(DefaultNumFactorList)}},
                       "合成方式":{"type":"SingleOption","order":1,"range":["直接合成","归一合成"]},
                       "缺失处理":{"type":"SingleOption","order":2,"range":["填充None","剩余合成"]}}
            return QSArgs(Args, ArgInfo, None)
        args._QS_MonitorChange = False
        if not set(args["描述子"].index).issubset(set(DefaultNumFactorList)):
            args["描述子"] = pd.DataFrame([("降序",1.0)],index=[DefaultNumFactorList[0]],columns=["排序方向", "合成权重"])
        args.ArgInfo["描述子"]["row"] = DefaultNumFactorList
        args.ArgInfo["描述子"]["unchecked_value"] = {"排序方向":"降序","合成权重":1/len(DefaultNumFactorList)}
        args._QS_MonitorChange = True
        return args
    # 产生筛选参数
    def _genFilterArgs(self, args, ds_name):
        DS = self.QSEnv.DSs[ds_name]
        if args is None:
            Args = {"信号类型":"多头信号",
                    "筛选条件":None,
                    "合并因子":self._genFactorMergeArgs(None, ds_name),
                    "排序方向":"降序",
                    "筛选方式":"定量",
                    "筛选数目":30,
                    "中性因子":[],
                    "类别过滤":None,
                    "换手缓冲":self._genTurnoverBufferArgs(None)}
            ArgInfo = {"信号类型":{"type":"SingleOption","order":0,"range":['多头信号','空头信号']},
                       "筛选条件":{"type":"IDFilter","order":1,"factor_list":DS.FactorNames},
                       "合并因子":{"type":"ArgSet", "order":2},
                       "排序方向":{"type":"SingleOption","order":3,"range":["降序","升序"]},
                       "筛选方式":{"type":"SingleOption","order":4,"range":["定量","定比","定比&定量"],"refresh":True},
                       "筛选数目":{"type":"Integer","order":5,"min":0,"max":np.inf,"single_step":1},
                       "中性因子":{"type":"MultiOption","order":6,"range":DS.FactorNames},
                       "类别过滤":{"type":"IDFilter","order":7,"factor_list":DS.FactorNames},
                       "换手缓冲":{"type":"ArgSet","order":8}}
            return QSArgs(Args, ArgInfo, self._onFilterArgChanged)
        args._QS_MonitorChange = False
        args["合并因子"] = self._genFactorMergeArgs(args["合并因子"], ds_name)
        args.ArgInfo["筛选条件"]["factor_list"] = DS.FactorNames
        args.ArgInfo["类别过滤"]["factor_list"] = DS.FactorNames
        args.ArgInfo["中性因子"]["range"] = DS.FactorNames
        if not set(args["中性因子"]).issubset(DS.FactorNames):
            args["中性因子"] = []
        args._QS_MonitorChange = True
        return args
    def _onFilterArgChanged(self, change_type, change_info, **kwargs):
        Args, Key, Value = change_info
        if (change_type=="set") and (Key=="筛选方式"):# 筛选方式发生了变化
            Args["筛选方式"] = Value
            if Value=='定量':
                Args.pop("筛选上限",None)
                Args.ArgInfo.pop("筛选上限",None)
                Args.pop("筛选下限",None)
                Args.ArgInfo.pop("筛选下限",None)
                Args["筛选数目"] = Args.get("筛选数目",30)
                Args.ArgInfo["筛选数目"] = {"type":"Integer","order":5,"min":0,"max":np.inf,"single_step":1}
            elif Value=='定比':
                Args.pop("筛选数目",None)
                Args.ArgInfo.pop("筛选数目",None)
                Args["筛选上限"] = Args.get("筛选上限",0.1)
                Args.ArgInfo["筛选上限"] = {"type":"Double","order":4.5,"min":0,"max":1,"single_step":0.0001}
                Args["筛选下限"] = Args.get("筛选下限",0.0)
                Args.ArgInfo["筛选下限"] = {"type":"Double","order":5.5,"min":0,"max":1,"single_step":0.0001}
            else:
                Args["筛选数目"] = Args.get("筛选数目",30)
                Args.ArgInfo["筛选数目"] = {"type":"Integer","order":4.5,"min":0,"max":np.inf,"single_step":1}
                Args["筛选上限"] = Args.get("筛选上限",0.1)
                Args.ArgInfo["筛选上限"] = {"type":"Double","order":5,"min":0,"max":1,"single_step":0.0001}
                Args["筛选下限"] = Args.get("筛选下限",0.0)
                Args.ArgInfo["筛选下限"] = {"type":"Double","order":5.5,"min":0,"max":1,"single_step":0.0001}
            return True
        return super(QSObject, self).__QS_onSysArgChanged__(change_type, change_info, **kwargs)
    # 产生权重分配参数
    def _genWeightAllocArgs(self, args, ds_name):
        if args is None:
            args = super()._genWeightAllocArgs(None, ds_name)
            args._QS_MonitorChange = False
            args['重配权重'] = True
            args.ArgInfo['重配权重']['readonly'] = True
            args._QS_MonitorChange = True
        return args
    def __QS_genSysArgs__(self, args=None, **kwargs):
        SysArgs = super().__QS_genSysArgs__(args, **kwargs)
        if self.QSEnv.DSs.isEmpty():
            return SysArgs
        DefaultDS = (self.QSEnv.DSs[args["数据源"]] if (args is not None) and (args.get("数据源") in self.QSEnv.DSs) else self.QSEnv.DSs.getDefaultDS())
        DefaultNumFactorList, DefaultStrFactorList = getFactorList(DefaultDS.DataType)
        if (args is None) or ("数据源" not in args):
            nSysArgs = len(SysArgs)
            SysArgs._QS_MonitorChange = False
            SysArgs.update({"筛选层数":1, "第0层":self._genFilterArgs(None, SysArgs["数据源"])})
            SysArgs.ArgInfo.update({"筛选层数":{"type":"Integer","order":nSysArgs,"min":0,"max":np.inf,"single_step":1,"refresh":True},
                                    "第0层":{"type":"ArgSet","order":nSysArgs+1}})
            SysArgs._QS_MonitorChange = True
            return SysArgs
        args = super().__QS_genSysArgs__(args=args)
        args._QS_MonitorChange = False
        for i in range(SysArgs["筛选层数"]):
            self._genFilterArgs(args=args["第"+str(i)+"层"], ds_name=args["数据源"])
        args._QS_MonitorChange = True
        return args
    def __QS_onSysArgChanged__(self, change_type, change_info, **kwargs):
        Args, Key, Value = change_info
        if (change_type=="set") and (Key=="筛选层数"):# 筛选方式发生了变化
            nArg = len(Args)
            nLevelChanged = int(Value+Args.ArgInfo["筛选层数"]["order"]-nArg)
            if nLevelChanged>0:
                for i in range(nLevelChanged):
                    iArg = "第"+str(int(Args["筛选层数"]+i))+"层"
                    Args[iArg] = self._genFilterArgs(Args["数据源"])
                    Args.ArgInfo[iArg] = {"type":"ArgSet","order":nArg+i}
            elif nLevelChanged<0:
                for i in range(abs(nLevelChanged)):
                    iArg = "第"+str(int(Args["筛选层数"]-i-1))+"层"
                    Args.pop(iArg,None)
                    Args.ArgInfo.pop(iArg,None)
            Args["筛选层数"] = Value
            return True
        return super().__QS_onSysArgChanged__(change_type, change_info, **kwargs)
    # ID过滤函数
    def _filtrateID(self, idt, ids, args):
        if args["筛选条件"] is not None:
            IDMask = self._DS.getIDMask(idt=idt, is_filtered=True, id_filter_str=args['筛选条件'])[ids]
            ids = list(IDMask[IDMask].index)
        DescriptorData = self._DS.getDateTimeData(idt=idt,ids=ids,factor_names=list(args["合并因子"]['描述子'].index))
        nID = DescriptorData.shape[0]
        TotalPosWeight = args['合并因子']["描述子"]["合成权重"].sum()
        MergeFactor = pd.Series(np.zeros(nID),index=DescriptorData.index)
        if args['合并因子']['缺失处理']=='剩余合成':
            TotalWeight = pd.Series(np.ones(nID),index=DescriptorData.index)*TotalPosWeight
            for i,iDescriptor in enumerate(args['合并因子']['描述子'].index):
                NullInd = pd.isnull(DescriptorData.loc[:,iDescriptor])
                TotalWeight -= NullInd*np.abs(args['合并因子']["描述子"]['合成权重'].iloc[i])
                temp = DescriptorData.loc[:,iDescriptor]
                temp[NullInd] = 0
                MergeFactor += temp*args['合并因子']["描述子"]['合成权重'].iloc[i]*(1 if args['合并因子']["描述子"]['排序方向'].iloc[i]=="降序" else -1)
            if args['合并因子']['合成方式']=='归一合成':
                MergeFactor = MergeFactor/TotalWeight
        elif args['合并因子']['缺失处理']=='填充None':
            for i,iDescriptor in enumerate(args['合并因子']['描述子'].index):
                iDescriptorData = DescriptorData.loc[:,iDescriptor].astype('float')
                MergeFactor += iDescriptorData*args['合并因子']["描述子"]['合成权重'].iloc[i]*(1 if args['合并因子']["描述子"]['排序方向'].iloc[i]=="降序" else -1)
            if args['合并因子']['合成方式']=='归一合成':
                MergeFactor = MergeFactor/TotalPosWeight
        MergeFactor = MergeFactor[pd.notnull(MergeFactor)].copy()
        if args['排序方向']=='降序':
            MergeFactor = -MergeFactor
        MergeFactor.sort_values(inplace=True,ascending=True)
        if args['筛选方式']=='定比':
            UpLimit = MergeFactor.quantile(args['筛选上限'])
            DownLimit = MergeFactor.quantile(args['筛选下限'])
            NewIDs = list(MergeFactor[(MergeFactor>=DownLimit) & (MergeFactor<=UpLimit)].index)
        elif args['筛选方式']=='定量':
            NewIDs = list(MergeFactor.iloc[:args['筛选数目']].index)
        elif args['筛选方式']=='定比&定量':
            UpLimit = MergeFactor.quantile(args['筛选上限'])
            DownLimit = MergeFactor.quantile(args['筛选下限'])
            NewIDs = set(MergeFactor[(MergeFactor>=DownLimit) & (MergeFactor<=UpLimit)].index)
            NewIDs = list(set(MergeFactor.iloc[:args['筛选数目']].index).intersection(NewIDs))
        if not args['换手缓冲']['是否缓冲']:
            return NewIDs
        SignalIDs = set(NewIDs)
        nSignalID = len(SignalIDs)
        if args["信号类型"]=="多头信号":
            if self.AllLongSignals=={}:
                LastIDs = set()
            else:
                LastIDs = set(self.AllLongSignals[max(list(self.AllLongSignals.keys()))].index)
        else:
            if self.AllShortSignals=={}:
                LastIDs = set()
            else:
                LastIDs = set(self.AllShortSignals[max(list(self.AllShortSignals.keys()))].index)
        if args['筛选方式']=='定比':
            UpLimit = MergeFactor.quantile(min((1.0,args['筛选上限']+args['换手缓冲']['筛选上限缓冲区'])))
            DownLimit = MergeFactor.quantile(max((0.0,args['筛选下限']-args['换手缓冲']['筛选下限缓冲区'])))
            NewIDs = LastIDs.intersection(set(MergeFactor[(MergeFactor>=DownLimit) & (MergeFactor<=UpLimit)].index))
        elif args['筛选方式']=='定量':
            NewIDs = LastIDs.intersection(set(MergeFactor.iloc[:args['筛选数目']+args['换手缓冲']['筛选数目缓冲区']].index))
        elif args['筛选方式']=='定比&定量':
            UpLimit = MergeFactor.quantile(min((1.0,args['筛选上限']+args['换手缓冲']['筛选上限缓冲区'])))
            DownLimit = MergeFactor.quantile(max((0.0,args['筛选下限']-args['换手缓冲']['筛选下限缓冲区'])))
            NewIDs = set(MergeFactor[(MergeFactor>=DownLimit) & (MergeFactor<=UpLimit)].index)
            NewIDs = LastIDs.intersection(set(MergeFactor.iloc[:args['筛选数目']+args['换手缓冲']['筛选数目缓冲区']].index).intersection(NewIDs))
        if len(NewIDs)>=nSignalID:# 当前持有的股票已经满足要求
            MergeFactor = MergeFactor[list(NewIDs)].copy()
            MergeFactor.sort_values(inplace=True,ascending=True)
            return list(MergeFactor.iloc[:nSignalID].index)
        SignalIDs = list(SignalIDs.difference(NewIDs))
        MergeFactor = MergeFactor[SignalIDs].copy()
        MergeFactor.sort_values(inplace=True,ascending=True)
        return list(NewIDs)+list(MergeFactor.iloc[:(nSignalID-len(NewIDs))].index)
    # 生成信号ID
    def _genSignalIDs(self, idt, original_ids, signal_type):
        IDs = original_ids
        for i in range(self.SysArgs['筛选层数']):
            iArgs = self.SysArgs["第"+str(i)+"层"]
            if iArgs['信号类型']!=signal_type:
                continue
            isNone = False
            if iArgs['中性因子']!=[]:
                if (iArgs['类别过滤'] is not None) and (iArgs['类别过滤']!=''):
                    iIDs = self._DS.getID(idt=idt, is_filtered=True, id_filter_str=iArgs['类别过滤'])
                    IDs = list(set(iIDs).intersection(set(IDs)))
                    IDs.sort()
                ClassData = self._DS.getDateTimeData(idt=idt,ids=IDs,factor_names=iArgs['中性因子'])
                if ClassData.shape[0]>0:
                    ClassData[pd.isnull(ClassData)] = np.nan
                AllClasses = [list(ClassData[iClass].unique()) for iClass in iArgs['中性因子']]
                AllClasses = CartesianProduct(AllClasses)
                IDs = []
                for jClass in AllClasses:
                    if pd.notnull(jClass[0]):
                        jMask = (ClassData[iArgs['中性因子'][0]]==jClass[0])
                    else:
                        jMask = pd.isnull(ClassData[iArgs['中性因子'][0]])
                    for k,kSubClass in enumerate(jClass[1:]):
                        if pd.notnull(kSubClass):
                            jMask = jMask & (ClassData[iArgs['中性因子'][k+1]]==kSubClass)
                        else:
                            jMask = jMask & pd.isnull(ClassData[iArgs['中性因子'][k+1]])
                    jIDs = self._filtrateID(idt,list(ClassData[jMask].index),iArgs)
                    IDs += jIDs
            else:
                IDs = self._filtrateID(idt,IDs,iArgs)
        return IDs
    def genLongSignal(self, idt, timestamp, trading_record):
        if self.SysArgs["多头账户"]=="无":
            return None
        OriginalIDs = self._DS.getID(idt, is_filtered=True)
        IDs = self._genSignalIDs(idt, OriginalIDs, '多头信号')
        return self._allocateWeight(idt, IDs, OriginalIDs, self.SysArgs['多头权重配置'])
    def genShortSignal(self, idt, timestamp, trading_record):
        if self.SysArgs["空头账户"]=="无":
            return None
        OriginalIDs = self._DS.getID(idt, is_filtered=True)
        IDs = self._genSignalIDs(idt, OriginalIDs, '空头信号')
        Signal = self._allocateWeight(idt, IDs, OriginalIDs, self.SysArgs['空头权重配置'])
        if Signal is not None:
            return -Signal
        else:
            return None
