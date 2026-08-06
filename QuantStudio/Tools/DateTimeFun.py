# coding=utf-8
"""日期和时间的操作函数"""
import datetime as dt
from typing import List, Literal, Union

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
def lookbackDateTime(idt: dt.datetime, period:str) -> dt.datetime:
    """回溯时点

    Args:
        idt: 回溯的起始时点
        period: 回溯期, 比如 '1m' 近一月, 将返回上个月的同日; '2w' 近2周, 将返回上两周的周末; 'monthsince' 表示本月以来, 将返回上个月月底
    
    Returns:
        回溯到的时点
    """
    Period = period.lower()
    if Period.endswith("since"):
        if Period == "weeksince":# 本周以来
            return idt - dt.timedelta(idt.weekday() + 1)
        elif Period == "monthsince":# 本月以来
            return dt.datetime(idt.year, idt.month, 1) - dt.timedelta(1)
        elif Period == "quartersince":# 本季以来
            return dt.datetime(idt.year, (1, 4, 7, 10)[idt.month // 3], 1) - dt.timedelta(1)
        elif Period == "yearsince":# 本年以来
            return dt.datetime(idt.year, 1, 1) - dt.timedelta(1)
        else:
            raise Exception(f"无法识别的 period: {period}")
    else:
        n, freq = int(Period[:-1]), Period[-1]
        if freq == "w":
            return idt - dt.timedelta(7 * n)
        elif freq == "m":
            try:
                return dt.datetime(idt.year - n // 12 - int(idt.month < n % 12), idt.month + int(idt.month < n % 12) * 12 - n % 12, idt.day)
            except:
                n = n - 1
                return dt.datetime(idt.year - n // 12 - int(idt.month < n % 12), idt.month + int(idt.month < n % 12) * 12 - n % 12, 1) - dt.timedelta(1)
        elif freq == "q":
            n = n * 3
            try:
                return dt.datetime(idt.year - n // 12 - int(idt.month < n % 12), idt.month + int(idt.month < n % 12) * 12 - n % 12, idt.day)
            except:
                n = n - 1
                return dt.datetime(idt.year - n // 12 - int(idt.month < n % 12), idt.month + int(idt.month < n % 12) * 12 - n % 12, 1) - dt.timedelta(1)
        elif freq == "y":
            try:
                return dt.datetime(idt.year - n, idt.month, idt.day)
            except:
                return dt.datetime(idt.year - n, 2, 28)
        else:
            raise Exception(f"无法识别的 period: {period}")


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
# 获取某个时点序列的月度时点序列
# exact=False: 是否精确的取目标 monthday
# postpone=True: 取每月大于等于 target_day 的第一个时点
# postpone=False: 取每月小于等于 target_day 的最后一个时点
# over_month=True: 表示允许跨月顺延
def getMonthDateTime(dts, target_day=15, exact=False, postpone=True, over_month=False):
    if exact: return [iDT for iDT in sorted(dts) if iDT.day==target_day]
    if over_month:
        dts = np.array(sorted(dts), dtype="O")
        DTStrs = [iDT.strftime("%Y%m%d") for iDT in dts]
        StartOffset = (1 if (not postpone) and (dts[0].day>target_day) else 0)
        EndOffset = (1 if postpone and (dts[-1].day<target_day) else 0)
        StartYear, StartMonth = dts[0].year, dts[0].month
        EndYear, EndMonth = dts[-1].year, dts[-1].month
        nMonth = (EndYear - StartYear) * 12 + EndMonth - StartMonth + 1
        NaturalDTStrs = []
        for i in range(StartOffset, nMonth-EndOffset):
            iYearNum, iMonthNum = i//12, i%12
            iTargetYear = StartYear + iYearNum
            iTargetMonth = StartMonth + iMonthNum
            iTargetYear += (iTargetMonth>12)
            iTargetMonth -= (iTargetMonth>12)*12
            NaturalDTStrs.append(str(iTargetYear)+str(iTargetMonth).zfill(2)+str(target_day).zfill(2))
        if postpone:
            return sorted(set(dts[np.searchsorted(DTStrs, NaturalDTStrs, side="left")]))
        else:
            return sorted(set(dts[np.searchsorted(DTStrs, NaturalDTStrs, side="right")-1]))
    TargetDTs = []
    if postpone:
        for iDT in sorted(dts):
            if (iDT.day>=target_day) and ((not TargetDTs) or (iDT.year!=TargetDTs[-1].year) or (iDT.month!=TargetDTs[-1].month)):
                TargetDTs.append(iDT)
    else:
        for iDT in sorted(dts):
            if iDT.day<=target_day:
                if (not TargetDTs) or (iDT.year!=TargetDTs[-1].year) or (iDT.month!=TargetDTs[-1].month):
                    TargetDTs.append(iDT)
                else:
                    TargetDTs[-1] = iDT
    return TargetDTs
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
# 获取某个时点序列的周度时点序列
# exact=False: 是否精确的取目标 weekday
# postpone=True: 向后顺延, 取每周大于等于 target_weekday 的第一个时点
# postpone=False: 向前顺延, 取每周小于等于 target_weekday 的最后一个时点
# over_week=True: 表示允许跨周顺延
def getWeekDateTime(dts, target_weekday=3, exact=False, postpone=True, over_week=False):
    target_weekday -= 1
    if exact: return [iDT for iDT in sorted(dts) if iDT.weekday()==target_weekday]
    if over_week:
        dts = np.array(sorted(dts), dtype="O")
        if not postpone:
            StartDT = dts[0] + dt.timedelta(target_weekday - dts[0].weekday()+7 * (dts[0].weekday()>target_weekday))
        else:
            StartDT = dts[0] + dt.timedelta(target_weekday-dts[0].weekday())
        NaturalDTs = getDateTimeSeries(StartDT, dts[-1], timedelta=dt.timedelta(7))
        if postpone:
            return sorted(set(dts[np.searchsorted(dts, NaturalDTs, side="left")]))
        else:
            return sorted(set(dts[np.searchsorted(dts, NaturalDTs, side="right")-1]))
    TargetDTs = []
    if postpone:
        for iDT in sorted(dts):
            if (iDT.weekday()>=target_weekday) and ((not TargetDTs) or ((iDT.date()-TargetDTs[-1].date()).days != iDT.weekday()-TargetDTs[-1].weekday())):
                TargetDTs.append(iDT)
    else:
        for iDT in sorted(dts):
            if iDT.weekday()<=target_weekday:
                if (not TargetDTs) or ((iDT.date()-TargetDTs[-1].date()).days != iDT.weekday()-TargetDTs[-1].weekday()):
                    TargetDTs.append(iDT)
                else:
                    TargetDTs[-1] = iDT
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
def getDateTimeSeries(start_dt, end_dt, timedelta=dt.timedelta(1)):
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

def transformDateTime(dts: List[dt.datetime], freq:str="1m", target_day:Union[Literal["last", "first"], int]="last", exact:bool=False, postpone:bool=True) -> List[dt.datetime]:
    """从给定的时点序列根据规则转换为特定的时点序列
    
    Args:
        dts: 原始时点序列
        freq: 转换频率，数字+单位的格式，单位有：d(日), w(周), m(月), q(季), y(年)
        target_day: 每个周期里取的目标时点，比如 freq=1m, target_day=15 表示取每月的15日
        exact: 是否要精确的取目标时点，True 表示 dts 中如果不存在目标时点则该周期不取时点
        postpone: 当 exact 为 False 时是否向后顺延，True 向后顺延, 取每个周期大于等于 target_day 的第一个时点，False 向前顺延, 取每个周期小于等于 target_day 的最后一个时点
    """
    raise NotImplementedError


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
    # DTs = pd.date_range(dt.datetime(2018,1,1), dt.datetime(2019,12,30), freq="D")
    # s = pd.Series(np.random.randn(DTs.shape[0]), index=DTs)
    # df = groupbyYear(s)
    # print(df.head())
    # 测试 lookbackDateTime
    print(lookbackDateTime(dt.datetime(2025, 3, 31), period="13m"))

    print("===")