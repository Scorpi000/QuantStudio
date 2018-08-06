# coding=utf-8

DSTableFactor = {"ElementaryFactor":[("日收益率","日收益率"),("总市值","市值")],
                 "BarraDescriptor":[("ESTU","ESTU"),("Industry","Industry")],
                 "BarraFactor":None}

DSSysArgs = {"最大缓冲因子数":60,
             "向前缓冲日期数":630,
             "向后缓冲日期数":1}

ModelArgs = {"运行模式":"多进程",
             "子进程数":7,
             "因子数据库":"FactorDB",
             "收益率因子":"日收益率",
             "ESTU因子":"ESTU",
             "行业因子":"Industry",
             "所有行业":["Energy","Chemicals","Construction Materials","Diversified Metals","Materials",
                        "Aerospace and Defense","Building Products","Construction and Engineering",
                        "Electrical Equipment","Industrial Conglomerates","Industrial Machinery",
                        "Trading Companies and Distributors","Commercial and Professional Services",
                        "Airlines","Marine","Road Rail and Transportation Infrastructure","Automobiles and Components",
                        "Household Durables (non-Homebuilding)","Leisure Products Textiles Apparel and Luxury",
                        "Hotels Restaurants and Leisure","Media","Retail","Food Staples Retail Household Personal Prod",
                        "Beverages","Food Products","Health","Banks","Diversified Financial Services","Real Estate",
                        "Software","Hardware and Semiconductors","Utilities"],
             "风格因子":['Size', 'Beta', 'Momentum', 'ResidualVolatility', 'NonlinearSize', 
                        'BookToPrice', 'Liquidity', 'EarningsYield', 'Growth', 'Leverage'],
             "市值因子":"市值",
             "EigenfactorRiskAdjustment":False,
             "FactorVolatilityRegimeAdjustment":False,
             "BayesianShrinkage":False,
             "SpecificVolatilityRegimeAdjustment":False}

FactorCovESTArgs = {"预测期数":21,
                    "样本长度":720,
                    "自相关滞后期":10,
                    "相关系数半衰期":480,
                    "波动率半衰期":90}

SpecificRiskESTArgs = {"预测期数":21,
                       "样本长度":360,
                       "自相关滞后期":10,
                       "波动率半衰期":90,
                       "结构化模型回归风格因子":['Momentum', 'ResidualVolatility', 'Liquidity']}

EigenfactorRiskAdjustmentArgs = {"MonteCarlo次数":1000,
                                 "模拟日期长度":480,
                                 "拟合忽略样本数":9,
                                 "a":1.4}

FactorVolatilityRegimeAdjustmentArgs = {"样本长度":-1,
                                        "半衰期":90,
                                        "预测期数":-1}

BayesianShrinkageArgs = {"分组数":10,
                         "q":0.1}

SpecificVolatilityRegimeAdjustmentArgs = {"样本长度":-1,
                                          "半衰期":90,
                                          "预测期数":-1}

SaveArgs = {"风险数据库":"FRDB",
            "风险数据表":"BarraRiskData"}