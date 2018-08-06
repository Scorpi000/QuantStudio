# coding=utf-8

DSTableFactor = {"ZXIElementaryFactor":[("日收益率","日收益率")]}

DSSysArgs = {"最大缓冲因子数":60,
             "向前缓冲日期数":630,
             "向后缓冲日期数":1}

ModelArgs = {"运行模式":"串行",
             "子进程数":8,
             "因子数据库":"FactorDB",
             "收益率因子":"日收益率",
             "ID过滤条件":None}

CovESTArgs = {"预测期数":21,
              "样本长度":720,
              "自相关滞后期":10,
              "相关系数半衰期":480,
              "波动率半衰期":90}

SaveArgs = {"风险数据库":"RDB",
            "风险数据表":"SampleEstZXIRiskData"}