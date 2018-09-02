# coding=utf-8
"""日期和时间的操作函数"""
import datetime as dt

import numpy as np
import pandas as pd

# 日期字符串(20120202)转成datetime(timestamp)，如果不是日期字符串，则返回None
def DateStr2Datetime(date_str):
    try:
        return pd.datetime(int(date_str[0:4]),int(date_str[4:6]),int(date_str[6:8]))
    except:
        return None
# datetime(timestamp)转成日期字符串(20120202)
def Datetime2DateStr(date):
    Year = date.year
    Month = date.month
    Day = date.day
    if Month<10:
        Month = '0'+str(Month)
    else:
        Month = str(Month)
    if Day<10:
        Day = '0'+str(Day)
    else:
        Day = str(Day)
    return str(Year)+Month+Day
# 截取日期序列
def cutDate(dates, start_date=None, end_date=None):
    if (start_date is None) and (end_date is None):
        return list(dates)
    dates = np.array(dates)
    if start_date is not None:
        dates = dates[dates>=start_date]
    if end_date is not None:
        dates = dates[dates<=end_date]
    return list(dates)
# 截取时点序列
def cutDateTime(dts, start_dt=None, end_dt=None):
    if (start_dt is None) and (end_dt is None): return list(dts)
    dts = np.array(dts)
    if start_dt is not None: dts = dts[dts>=start_dt]
    if end_dt is not None: dts = dts[dts<=end_dt]
    return dts.tolist()
# 获取某个日期序列的每月最后一天序列
def getMonthLastDay(dates):
    dates.sort()
    MonthLastDay = []
    MonthLastDay.append(dates[0])
    for iDate in dates:
        if (iDate[:6]==MonthLastDay[-1][:6]):
            MonthLastDay[-1] = iDate
        else:
            MonthLastDay.append(iDate)
    return MonthLastDay
# 获取某个日期序列的每月中间一天序列，每月小于等于15日的最后一天
def getMonthMiddleDay(dates):
    dates.sort()
    MonthMiddleDay = [dates[0]]
    for iDate in dates:
        if (iDate[:6]==MonthMiddleDay[-1][:6]):
            if int(iDate[-2:])<=15:
                MonthMiddleDay[-1] = iDate
        else:
            MonthMiddleDay.append(iDate)
    return MonthMiddleDay
# 获取某个日期序列的每月第一天序列
def getMonthFirstDay(dates):
    dates.sort()
    MonthFirstDay = []
    MonthFirstDay.append(dates[0])
    for iDate in dates:
        if (iDate[:6]!=MonthFirstDay[-1][:6]):
            MonthFirstDay.append(iDate)
    return MonthFirstDay
# 获取某个日期序列的每周最后一天序列
def getWeekLastDay(dates):
    dates.sort()
    WeekLastDay = []
    WeekLastDay.append(dates[0])
    for iDate in dates:
        iDateTime = DateStr2Datetime(iDate)
        iDateWeekDay = iDateTime.weekday()
        tempDateTime = DateStr2Datetime(WeekLastDay[-1])
        tempWeekDay = tempDateTime.weekday()
        if (iDateTime-tempDateTime).days != (iDateWeekDay-tempWeekDay):
            WeekLastDay.append(iDate)
        else:
            WeekLastDay[-1] = iDate
    return WeekLastDay
# 获取某个日期序列的每周第一天序列
def getWeekFirstDay(dates):
    dates.sort()
    WeekFirstDay = []
    WeekFirstDay.append(dates[0])
    for iDate in dates:
        iDateTime = DateStr2Datetime(iDate)
        iDateWeekDay = iDateTime.weekday()
        tempDateTime = DateStr2Datetime(WeekFirstDay[-1])
        tempWeekDay = tempDateTime.weekday()
        if (iDateTime-tempDateTime).days != (iDateWeekDay-tempWeekDay):
            WeekFirstDay.append(iDate)
    return WeekFirstDay
# 获取某个日期序列的每年最后一天序列
def getYearLastDay(dates):
    dates.sort()
    YearLastDay = []
    YearLastDay.append(dates[0])
    for iDate in dates:
        if (iDate[:4]==YearLastDay[-1][:4]):
            YearLastDay[-1] = iDate
        else:
            YearLastDay.append(iDate)
    return YearLastDay
# 获取某个日期序列的每年第一天序列
def getYearFirstDay(dates):
    dates.sort()
    YearFirstDay = []
    YearFirstDay.append(dates[0])
    for iDate in dates:
        if (iDate[:4]!=YearFirstDay[-1][:4]):
            YearFirstDay.append(iDate)
    return YearFirstDay
# 获取某个日期序列的每个季度第一天序列
def getQuarterFirstDay(dates):
    dates.sort()
    QuarterFirstDay = []
    QuarterFirstDay.append(dates[0])
    for iDate in dates:
        if (iDate[:4]!=QuarterFirstDay[-1][:4]):
            QuarterFirstDay.append(iDate)
        elif int((int(iDate[4:6])-1)/3)!=int((int(QuarterFirstDay[-1][4:6])-1)/3):
            QuarterFirstDay.append(iDate)
    return QuarterFirstDay
# 获取某个日期序列的每个季度最后一天序列
def getQuarterLastDay(dates):
    dates.sort()
    QuarterLastDay = []
    QuarterLastDay.append(dates[0])
    for iDate in dates:
        if (iDate[:4]!=QuarterLastDay[-1][:4]):
            QuarterLastDay.append(iDate)
        elif int((int(iDate[4:6])-1)/3)!=int((int(QuarterLastDay[-1][4:6])-1)/3):
            QuarterLastDay.append(iDate)
        else:
            QuarterLastDay[-1] = iDate
    return QuarterLastDay
# 获取某个日期序列的每个财报公布季度最后一天序列,上年11月初至当年四月底为第一季度，5月初至八月底为第二季度，9月初至十月底为第三季度
def getFinancialQuarterLastDay(dates):
    def getQuanterNum(date):
        if int(date[4:6]) in [1,2,3,4,11,12]:
            return 0
        elif int(date[4:6]) in [5,6,7,8]:
            return 2
        else:
            return 3
    dates.sort()
    QuarterLastDay = []
    QuarterLastDay.append(dates[0])
    for iDate in dates:
        if (iDate[:4]==QuarterLastDay[-1][:4]):
            if (getQuanterNum(iDate)!=getQuanterNum(QuarterLastDay[-1])) or ((int(iDate[4:6])>=11) and (int(QuarterLastDay[-1][4:6])<=4)):
                QuarterLastDay.append(iDate)
            else:
                QuarterLastDay[-1] = iDate
        elif (int(iDate[:4])-int(QuarterLastDay[-1][:4])>1):
            QuarterLastDay.append(iDate)
        else:
            if (int(QuarterLastDay[-1][4:6])>=11) and (int(iDate[4:6])<=4):
                QuarterLastDay[-1] = iDate
            elif getQuanterNum(iDate)!=getQuanterNum(QuarterLastDay[-1]):
                QuarterLastDay.append(iDate)
            else:
                QuarterLastDay[-1] = iDate
    return QuarterLastDay
# 获取某个日期序列的每个财报公布季度最后一天序列,上年11月初至当年四月底为第一季度，5月初至八月底为第二季度，9月初至十月底为第三季度
def getFinancialQuarterFirstDay(dates):
    def getQuanterNum(date):
        if int(date[4:6]) in [1,2,3,4,11,12]:
            return 0
        elif int(date[4:6]) in [5,6,7,8]:
            return 2
        else:
            return 3
    dates.sort()
    QuarterFirstDay = []
    QuarterFirstDay.append(dates[0])
    for iDate in dates:
        if (iDate[:4]==QuarterFirstDay[-1][:4]):# 同一年
            if (getQuanterNum(iDate)!=getQuanterNum(QuarterFirstDay[-1])) or ((int(iDate[4:6])>=11) and (int(QuarterFirstDay[-1][4:6])<=4)):
                QuarterFirstDay.append(iDate)
        elif (int(iDate[:4])-int(QuarterFirstDay[-1][:4])>1):# 相差超过一年
            QuarterFirstDay.append(iDate)
        else:
            if getQuanterNum(iDate)!=getQuanterNum(QuarterFirstDay[-1]):
                QuarterFirstDay.append(iDate)
    return QuarterFirstDay
# 日期变换
def changeDate(dates, change_type=None):
    if change_type is None:
        return ['月末日',"周末日","年末日","季末日","月初日","周初日","年初日","季初日","A股财报季初日","A股财报季末日","月中日"]
    if change_type == '月末日':
        return getMonthLastDay(dates)
    elif change_type == '周末日':
        return getWeekLastDay(dates)
    elif change_type == '年末日':
        return getYearLastDay(dates)
    elif change_type == '季末日':
        return getQuarterLastDay(dates)
    elif change_type == '月初日':
        return getMonthFirstDay(dates)
    elif change_type == '周初日':
        return getWeekFirstDay(dates)
    elif change_type == '年初日':
        return getYearFirstDay(dates)
    elif change_type == '季初日':
        return getQuarterFirstDay(dates)
    elif change_type == '财报季初日':
        return getFinancialQuarterFirstDay(dates)
    elif change_type == '财报季末日':
        return getFinancialQuarterLastDay(dates)
    elif change_type == '月中日':
        return getMonthMiddleDay(dates)
# 获取连续的自然日序列
def getNaturalDay(start_date,end_date):
    if start_date>end_date:
        return []
    Dates = []
    iDate = start_date
    while iDate<=end_date:
        Dates.append(iDate)
        iDate = Datetime2DateStr(DateStr2Datetime(iDate)+dt.timedelta(days=1))
    return Dates
# 获取当前日期的字符串形式
def getCurrentDateStr():
    Today = pd.datetime.today()
    return Datetime2DateStr(Today)
# 获取去N年的同一天，如果是2月29号，返回2月28号
def getLastNYearDate(date,n_year=1):
    Year = int(date[:4])-n_year
    if (date[4:]=='0229') and (Year%4!=0):
        return str(Year)+'0228'
    else:
        return str(Year)+date[4:]
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
# 获取某个时点序列的每月最后一个时点序列
def getMonthLastDateTime(dts):
    dts.sort()
    TargetDateTime = []
    TargetDateTime.append(dts[0])
    for iDateTime in dts:
        if (iDateTime.year==TargetDateTime[-1].year) and (iDateTime.month==TargetDateTime[-1].month):
            TargetDateTime[-1] = iDateTime
        else:
            TargetDateTime.append(iDateTime)
    return TargetDateTime
# 获取日期序列
def getDateSeries(start_date, end_date):
    return (start_date-dt.timedelta(1))+np.array([dt.timedelta(1)]*((end_date-start_date).days+1)).cumsum()
# 获取日内连续的时间序列, start_time, end_time, timedelta 是 datetime.time 对象
def getTimeSeries(start_time, end_time, timedelta):
    TimeSeries = getDateTimeSeries(dt.datetime.combine(dt.date.today(), start_time), dt.datetime.combine(dt.date.today(), end_time), timedelta)
    return np.array(tuple(map(lambda x: x.time(), TimeSeries)))
# 获取连续的时间点序列
def getDateTimeSeries(start_dt, end_dt, timedelta):
    nDelta = int((end_dt-start_dt)/timedelta)+1
    return (start_dt-timedelta)+np.array([timedelta]*nDelta).cumsum()
if __name__=="__main__":
    import time
    #DateTimes = list(pd.date_range(dt.datetime(2018,1,1,9,30), dt.datetime(2018,2,1,15), freq="min"))
    #Dates = list(pd.date_range(dt.date(2018,1,1), dt.date(2018,2,1), freq="D"))
    #Index = getDateStartEndIndex(DateTimes, Dates)
    #DateTimes = getDateTimeSeries(dt.datetime(2018,1,1,9,30), dt.datetime(2018,2,1,15), dt.timedelta(minutes=5))
    Dates = getDateSeries(dt.date(2018,1,1), dt.date(2018,1,3))
    Times = getTimeSeries(dt.time(9,30), dt.time(11,30), dt.timedelta(minutes=1))
    DateTimes = np.array(tuple(combineDateTime(Dates, Times)))
    #DateIndex = getDateStartEndIndex(DateTimes, Dates)
    #LastDateTimes = DateTimes[DateIndex[:,1]-1]
    #StartT = time.clock()
    #DateTimes = getDateTimeSeries(dt.datetime(2018,1,1,9,30), dt.datetime(2018,12,31,15), dt.timedelta(seconds=1))
    #print(time.clock()-StartT)
    pass