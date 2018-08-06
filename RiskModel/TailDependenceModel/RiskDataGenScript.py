# coding=utf-8
"""生成风险数据"""
from QSEnvironment import QSEnv

if __name__=='__main__':
    from DateTimeFun import getMonthLastDay
    MainQSEnv = QSEnv()
    MainQSEnv.start()
    
    import TailDependenceModel
    MainModel = TailDependenceModel.TailDependenceModel("MainModel",config_file="TailDependenceModelConfig.py",qs_env=MainQSEnv)
    
    #MainModel.setRiskESTDate(["20050131","20050228"])# debug
    Dates = MainQSEnv.FactorDB.getDate("ElementaryFactor","流通市值",start_date="20080101",end_date="20170531")
    MainModel.setRiskESTDate(getMonthLastDay(Dates))
    
    ErrorCode = MainModel.initInfo()
    if ErrorCode!=1:
        print(MainQSEnv.SysArgs['LastErrorMsg'])
        exit()
    MainModel.genData()
    MainQSEnv.close()