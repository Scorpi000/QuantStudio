# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from QSEnvironment import QSEnv
import DataSource
import RiskDataSource
import MeanVariancePC
from DateTimeFun import getMonthLastDay

if __name__=='__main__':
    # 构建数据源
    MainQSEnv = QSEnv()
    MainQSEnv.start()
    DS = DataSource.ParaMMAPCacheDSWithDateCons("MainDS",MainQSEnv.FactorDB,MainQSEnv)
    TableFactor = {"ElementaryFactor":[("复权收盘价","复权收盘价"),("月收益率","月收益率"),("是否在市","是否在市"),("特殊处理","特殊处理"),("涨跌停","涨跌停"),("交易状态","交易状态")],
                   "IndexConstituentFactor":[("沪深300成份权重","沪深300成份权重"),("中证500成份权重","中证500成份权重"),("上证50成份权重","上证50成份权重")],
                   "BarraFactor":None,
                   "BarraDescriptor":[("Industry","Industry")],
                   "StyleTechnicalFactor":[("AmountAvg_1M","AmountAvg_1M")]}
    DS.prepareData(TableFactor)
    DS.Dates = getMonthLastDay(DS.Dates)
    #DS.setIDFilter("pd.isnull(@特殊处理) & (@是否在市==1) & ((@交易状态==-1) | (@交易状态>0)) & (@涨跌停==0)")
    #DS.setIDFilter("((@中证500成份权重>0) | (@沪深300成份权重>0)) & (@是否在市==1)")
    #DS.setIDFilter("(pd.isnull(@特殊处理) & (@是否在市==1))")
    DS.setIDFilter("(@上证50成份权重>0) & (@是否在市==1)")
    
    # 构建风险数据源
    RiskDS = RiskDataSource.ParaMMAPCacheFRDS("MainRiskDS",MainQSEnv.FRDB,MainQSEnv)
    RiskDS.prepareData("BarraRiskData")
    
    # 设置优化目标和约束条件
    NeutralFactors = ["Size","Beta","BookToPrice","EarningsYield","Growth","Leverage","Liquidity","Momentum","NonlinearSize","ResidualVolatility"]
    NeutralIndustryFactors = ["Industry"]
    PC = MeanVariancePC.MatlabMVPC("MainPC",MainQSEnv)
    PC.setObject({"收益项系数":0.0,"风险厌恶系数":1.0,"相对基准":True,'换手惩罚系数':0.0,"买入惩罚系数":0.0,"卖出惩罚系数":0.0})
    PC.addConstraint("预算约束",{"限制上限":1.0,"限制下限":1.0,"相对基准":False,"舍弃优先级":-1})
    PC.addConstraint("权重约束",{"目标ID":None,"限制上限":1.0,"限制下限":0.0,"相对基准":False,"舍弃优先级":-1})
    #PC.addConstraint("权重约束",{"目标ID":"@中证500成份权重>0","限制上限":0.02,"限制下限":0.0,"相对基准":False,"舍弃优先级":-1})
    #PC.addConstraint("因子暴露约束",{"因子类型":"数值型","因子名称":NeutralFactors,"相对基准":True,"限制上限":0.0001,"限制下限":-0.0001})
    #PC.addConstraint("因子暴露约束",{"因子类型":"类别型","因子名称":NeutralIndustryFactors,"相对基准":True,"限制上限":0.0001,"限制下限":-0.0001})
    #PC.addConstraint("换手约束",{"限制类型":"总换手限制","成交额倍数":1.0,"限制上限":0.7,"舍弃优先级":0})
    #PC.addConstraint("换手约束",{"限制类型":"买卖限制","成交额倍数":0.1,"限制上限":0.7,"舍弃优先级":0})
    #PC.addConstraint("波动率约束",{"限制上限":0.029,"相对基准":True,"舍弃优先级":-1})
    #PC.addConstraint("非零数目约束",{"限制上限":10,"相对基准":False,"舍弃优先级":-1})
    PC.setOptionArg({"信息显示":"Default","求解器":"Default"})
    
    # 初始化
    PC.initPC()
    
    # 设置相关数据
    TargetDate = "20161230"
    TargetIDs = DS.getID(TargetDate,True)
    PC.setTargetID(TargetIDs)
    if PC.UseExpectedReturn:
        PC.setExpectedReturn(DS.getDateTimeData(TargetDate,factor_names=["月收益率"])["月收益率"])
    if PC.UseCovMatrix:
        PC.setCovMatrix(factor_cov=RiskDS.getDateFactorCov(TargetDate),specific_risk=RiskDS.getDateSpecificRisk(TargetDate),
                        risk_factor_data=RiskDS.getDateFactorData(TargetDate))
    if PC.UseFactor!=[]:
        PC.setFactorData(factor_data=DS.getDateTimeData(idt=TargetDate,factor_names=UseFactor,ids=None))
    if PC.UseAmount:
        PC.setAmountData(DS.getDateTimeData(TargetDate,factor_names=["AmountAvg_1M"])["AmountAvg_1M"])
    if PC.UseHolding or PC.UseAmount or PC.UseWealth:
        #Holding = DS.getDateTimeData(TargetDate,factor_names=["中证500成份权重"])["中证500成份权重"]
        #Holding = Holding[Holding>0]
        Holding = pd.Series(0.0,index=TargetIDs)
        PC.setHolding(Holding)
        PC.setWealth(1000000000)# 当前的财富值
    if PC.UseBenchmark:
        PC.setBenchmarkHolding(DS.getDateTimeData(TargetDate,factor_names=["上证50成份权重"])["上证50成份权重"])# 当前的基准投资组合
    PC.setFilteredID(DS,TargetDate)
    
    # 求解优化问题
    Portfolio,ResultInfo = PC.solve()
    PC.endPC()
    
    MainQSEnv.close()