# coding=utf-8

DSTableFactor = {"ZXIElementaryFactor":[("日收益率","日收益率")]}

DSSysArgs = {"最大缓冲因子数":60,
             "向前缓冲日期数":630,
             "向后缓冲日期数":1}

ModelArgs = {"运行模式":"串行",
             "子进程数":8,
             "因子数据库":"FactorDB",
             "收益率因子":"日收益率",
             "ID过滤条件":None,
             "波动率来源表":"SampleEstZXIRiskData"}

TailDependenceArgs = {"样本长度":720}

SaveArgs = {"风险数据库":"RDB",
            "风险数据表":"TailDependenceZXIRiskData"}