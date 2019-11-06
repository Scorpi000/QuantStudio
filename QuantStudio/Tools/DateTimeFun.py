# coding=utf-8
"""日期和时间的操作函数"""
import datetime as dt

import numpy as np
import pandas as pd

# 截取日期序列, depreciated
def cutDate(dates, start_date=None, end_date=None):
    if (start_date is None) and (end_date is None):
        return list(dates)
    dates = np.array(dates)
    if start_date is not None:
        dates = dates[dates>=start_date]
    if end_date is not None:
        dates = dates[dates<=end_date]
    return list(dates)
# 截取时点序列, depreciated
def cutDateTime(dts, start_dt=None, end_dt=None):
    if (start_dt is None) and (end_dt is None): return list(dts)
    dts = np.array(dts)
    if start_dt is not None: dts = dts[dts>=start_dt]
    if end_dt is not None: dts = dts[dts<=end_dt]
    return dts.tolist()
# 获取连续的自然日序列, depreciated
def getNaturalDay(start_date,end_date):
    if start_date>end_date:
        return []
    Dates = []
    iDate = start_date
    while iDate<=end_date:
        Dates.append(iDate)
        iDate += dt.timedelta(days=1)
    return Dates
# -------------------------------新的基于 DateTime 的日期时间函数---------------------
# 合并日期序列和时间序列, 形成 DateTime 序列, 生成器函数
def combineDateTime(dates, times):
    for iDate in dates:
        for jTime in times:
            yield dt.datetime.combine(iDate, jTime)
# 获取日期在时间点序列中的开始和结束索引, array((len(dates),2))
def getDateStartEndIndex(dts, dates):
    dts = np.array(dts)
    nDate = len(dates)
    Index = np.full((nDate, 2), 0, dtype=np.int64)
    StartTime = dt.time(0)
    EndTime = dt.time(23,59,59,999999)
    for i, iDate in enumerate(dates):
        iDateTime = dt.datetime.combine(iDate, StartTime)
        Index[i, 0] = dts.searchsorted(iDateTime)
        iDateTime = dt.datetime.combine(iDate, EndTime)
        iIndex = dts.searchsorted(iDateTime)
        if (iIndex>nDate-1) or (dts[iIndex]==iDateTime):
            Index[i, 1] = iIndex
        else:
            Index[i, 1] = iIndex-1
    return Index
# 获取某个时点序列的每月第一个时点序列
def getMonthFirstDateTime(dts):
    dts = sorted(dts)
    TargetDTs = [dts[0]]
    for iDT in dts:
        if (iDT.year!=TargetDTs[-1].year) or (iDT.month!=TargetDTs[-1].month):
            TargetDTs.append(iDT)
    return TargetDTs
# 获取某个时点序列的每月中间一时点序列, 每月小于等于 middle_day(默认 15) 的最后一天的最后一个时点
def getMonthMiddleDateTime(dts, middle_day=15):
    dts = sorted(dts)
    TargetDTs = [dts[0]]
    for iDT in dts:
        if (iDT.year==TargetDTs[-1].year) and (iDT.month==TargetDTs[-1].month):
            if iDT.day<=middle_day:
                TargetDTs[-1] = iDT
        else:
            TargetDTs.append(iDT)
    return TargetDTs
# 获取某个时点序列的每月最后一个时点序列
def getMonthLastDateTime(dts):
    dts = sorted(dts)
    TargetDTs = [dts[0]]
    for iDT in dts:
        if (iDT.year==TargetDTs[-1].year) and (iDT.month==TargetDTs[-1].month):
            TargetDTs[-1] = iDT
        else:
            TargetDTs.append(iDT)
    return TargetDTs
# 获取某个时点序列的每周第一个时点序列
def getWeekFirstDateTime(dts):
    dts = sorted(dts)
    TargetDTs = [dts[0]]
    for iDT in dts:
        if (iDT.date()-TargetDTs[-1].date()).days != (iDT.weekday()-TargetDTs[-1].weekday()):
            TargetDTs.append(iDT)
    return TargetDTs
# 获取某个时点序列的每周最后一个时点序列
def getWeekLastDateTime(dts):
    dts = sorted(dts)
    TargetDTs = [dts[0]]
    for iDT in dts:
        if (iDT.date()-TargetDTs[-1].date()).days != (iDT.weekday()-TargetDTs[-1].weekday()):
            TargetDTs.append(iDT)
        else:
            TargetDTs[-1] = iDT
    return TargetDTs
# 获取某个时点序列的每年第一天序列
def getYearFirstDateTime(dts):
    dts = sorted(dts)
    TargetDTs = [dts[0]]
    for iDT in dts:
        if iDT.year!=TargetDTs[-1].year:
            TargetDTs.append(iDT)
    return TargetDTs
# 获取某个时点序列的每年最后一个时点序列
def getYearLastDateTime(dts):
    dts = sorted(dts)
    TargetDTs = [dts[0]]
    for iDT in dts:
        if (iDT.year==TargetDTs[-1].year):
            TargetDTs[-1] = iDT
        else:
            TargetDTs.append(iDT)
    return TargetDTs
# 获取某个时点序列的每个季度第一个时点序列
def getQuarterFirstDateTime(dts):
    dts = sorted(dts)
    TargetDTs = [dts[0]]
    for iDT in dts:
        if (iDT.year!=TargetDTs[-1].year):
            TargetDTs.append(iDT)
        elif (iDT.month-1)//3 != (TargetDTs[-1].month-1)//3:
            TargetDTs.append(iDT)
    return TargetDTs
# 获取某个时点序列的每个季度最后一个时点序列
def getQuarterLastDateTime(dts):
    dts = sorted(dts)
    TargetDTs = [dts[0]]
    for iDT in dts:
        if (iDT.year!=TargetDTs[-1].year):
            TargetDTs.append(iDT)
        elif (iDT.month-1)//3 != (TargetDTs[-1].month-1)//3:
            TargetDTs.append(iDT)
        else:
            TargetDTs[-1] = iDT
    return TargetDTs
def _getQuanterNum(idt):
    if idt.month in (1,2,3,4,11,12): return 1
    elif idt.month in (5,6,7,8): return 2
    else: return 3
# 获取某个时点序列的每个财报公布季度第一个时点序列, 上年 11 月初至当年 4 月底为第一季度, 5 月初至 8 月底为第二季度, 9 月初至 10 月底为第三季度
def getFinancialQuarterFirstDateTime(dts):
    dts = sorted(dts)
    TargetDTs = [dts[0]]
    for iDT in dts:
        if (iDT.year==TargetDTs[-1].year):# 同一年
            if (_getQuanterNum(iDT)!=_getQuanterNum(TargetDTs[-1])) or ((iDT.month>=11) and (TargetDTs[-1].month<=4)):
                TargetDTs.append(iDT)
        elif iDT.year-TargetDTs[-1].year>1:# 相差超过一年
            TargetDTs.append(iDT)
        else:
            if _getQuanterNum(iDT)!=_getQuanterNum(TargetDTs[-1]):
                TargetDTs.append(iDT)
    return TargetDTs
# 获取某个时点序列的每个财报公布季度最后一个时点序列, 上年 11 月初至当年 4 月底为第一季度, 5 月初至 8 月底为第二季度, 9 月初至 10 月底为第三季度
def getFinancialQuarterLastDateTime(dts):
    dts = sorted(dts)
    TargetDTs = [dts[0]]
    for iDT in dts:
        if (iDT.year==TargetDTs[-1].year):
            if (_getQuanterNum(iDT)!=_getQuanterNum(TargetDTs[-1])) or ((iDT.month>=11) and (TargetDTs[-1].month<=4)):
                TargetDTs.append(iDT)
            else:
                TargetDTs[-1] = iDT
        elif iDT.year-TargetDTs[-1].year>1:
            TargetDTs.append(iDT)
        else:
            if (TargetDTs[-1].month>=11) and (iDT.month<=4):
                TargetDTs[-1] = iDT
            elif _getQuanterNum(iDT)!=_getQuanterNum(TargetDTs[-1]):
                TargetDTs.append(iDT)
            else:
                TargetDTs[-1] = iDT
    return TargetDTs

# 获取日期序列
def getDateSeries(start_date, end_date):
    return ((start_date-dt.timedelta(1)) + np.array([dt.timedelta(1)] * ((end_date-start_date).days+1)).cumsum()).tolist()
# 获取日内连续的时间序列, start_time, end_time, timedelta 是 datetime.time 对象
def getTimeSeries(start_time, end_time, timedelta):
    TimeSeries = getDateTimeSeries(dt.datetime.combine(dt.date.today(), start_time), dt.datetime.combine(dt.date.today(), end_time), timedelta)
    return list(map(lambda x: x.time(), TimeSeries))
# 获取连续的时间点序列
def getDateTimeSeries(start_dt, end_dt, timedelta):
    nDelta = int((end_dt-start_dt)/timedelta)+1
    return ((start_dt-timedelta)+np.array([timedelta]*nDelta).cumsum()).tolist()
# 时间序列按照年度分组
# s: Series(index=[datetime]) -> DataFrame(index=["%m-%d"], columns=["%Y"])
def groupbyYear(s):
    Year, MonthDay = [], []
    for iDT in s.index:
        Year.append(iDT.strftime("%Y"))
        MonthDay.append(iDT.strftime("%m-%d"))
    return pd.DataFrame({"Year": Year, "MonthDay": MonthDay, "Data": s.values}).set_index(["MonthDay", "Year"]).unstack()

if __name__=="__main__":
    import time
    #DateTimes = list(pd.date_range(dt.datetime(2018,1,1,9,30), dt.datetime(2018,2,1,15), freq="min"))
    #Dates = list(pd.date_range(dt.date(2018,1,1), dt.date(2018,2,1), freq="D"))
    #Index = getDateStartEndIndex(DateTimes, Dates)
    #DateTimes = getDateTimeSeries(dt.datetime(2018,1,1,9,30), dt.datetime(2018,2,1,15), dt.timedelta(minutes=5))
    #Dates = getDateSeries(dt.date(2018,1,1), dt.date(2018,1,3))
    #Times = getTimeSeries(dt.time(9,30), dt.time(11,30), dt.timedelta(minutes=1))
    #DateTimes = np.array(tuple(combineDateTime(Dates, Times)))
    #DateIndex = getDateStartEndIndex(DateTimes, Dates)
    #LastDateTimes = DateTimes[DateIndex[:,1]-1]
    #StartT = time.perf_counter()
    #DateTimes = getDateTimeSeries(dt.datetime(2018,1,1,9,30), dt.datetime(2018,12,31,15), dt.timedelta(seconds=1))
    #print(time.perf_counter()-StartT)
    # 测试 groupbyYear
    DTs = pd.date_range(dt.datetime(2018,1,1), dt.datetime(2019,12,30), freq="D")
    s = pd.Series(np.random.randn(DTs.shape[0]), index=DTs)
    df = groupbyYear(s)
    print(df.head())
    print("===")