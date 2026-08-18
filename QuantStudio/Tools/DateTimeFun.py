# coding=utf-8
"""日期和时间的操作函数"""
import datetime as dt
from typing import List, Literal, Union, Optional

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

# region: 统一的时点序列变换
def transformDateTime(dts: List[dt.datetime], freq:str="1m", target_day:Union[Literal["last", "first"], int]="last", exact:bool=False, postpone:bool=True, over_period:bool=True) -> List[dt.datetime]:
    """从给定的时点序列根据规则转换为特定的时点序列

    Args:
        dts: 原始时点序列
        freq: 转换频率，数字+单位的格式，单位有：d(日), w(周), m(月), q(季), y(年)
        target_day: 每个周期里取的目标时点，比如 freq=1m, target_day=15 表示取每月的15日
        exact: 是否要精确的取目标时点，True 表示 dts 中如果不存在目标时点则该周期不取时点
        postpone: 当 exact 为 False 时是否向后顺延，True 向后顺延, 取每个周期大于等于 target_day 的第一个时点，False 向前顺延, 取每个周期小于等于 target_day 的最后一个时点
        over_period: 表示是否允许跨周期顺延
    
    Returns:
        转换后的时点序列
    """
    freq = freq.lower()
    n = int(freq[:-1])
    unit = freq[-1]

    if unit not in ("d", "w", "m", "q", "y"):
        raise ValueError(f"不支持的频率单位: {unit}，支持的频率单位：d(日), w(周), m(月), q(季), y(年)")

    dts = sorted(dts)
    if not dts:
        return []

    if unit == "d":
        # 日频：按 target_day（跨步 n 天）采样
        if target_day == "last":
            return dts[::n] if n > 1 else list(dts)
        elif target_day == "first":
            return [dts[i] for i in range(0, len(dts), n)]
        else:
            # target_day 为整数时表示每 n 天取第 target_day 个（从周期起始算起，1-based）
            target_idx = target_day - 1
            dts_arr = np.array(dts, dtype="O")
            n_periods = len(dts) // n
            indices = list(range(0, n_periods * n, n))
            target_indices = [i + target_idx for i in indices if i + target_idx < len(dts)]
            return dts_arr[target_indices].tolist()

    if unit == "w":
        if target_day == "last":
            return _transform_by_period(dts, "w", n, None, exact, postpone, over_period, take_last=True)
        elif target_day == "first":
            return _transform_by_period(dts, "w", n, None, exact, postpone, over_period, take_first=True)
        else:
            # target_day 为周几（1=周一, 7=周日），转为 weekday() 体系（0=周一, 6=周日）
            return _transform_by_period(dts, "w", n, target_day - 1, exact, postpone, over_period)

    if unit == "m":
        if target_day == "last":
            return _transform_by_period(dts, "m", n, None, exact, postpone, over_period, take_last=True)
        elif target_day == "first":
            return _transform_by_period(dts, "m", n, None, exact, postpone, over_period, take_first=True)
        else:
            return _transform_by_period(dts, "m", n, target_day, exact, postpone, over_period)

    if unit == "q":
        if target_day == "last":
            return _transform_by_period(dts, "q", n, None, exact, postpone, over_period, take_last=True)
        elif target_day == "first":
            return _transform_by_period(dts, "q", n, None, exact, postpone, over_period, take_first=True)
        else:
            return _transform_by_period(dts, "q", n, target_day, exact, postpone, over_period)

    if unit == "y":
        if target_day == "last":
            return _transform_by_period(dts, "y", n, None, exact, postpone, over_period, take_last=True)
        elif target_day == "first":
            return _transform_by_period(dts, "y", n, None, exact, postpone, over_period, take_first=True)
        else:
            return _transform_by_period(dts, "y", n, target_day, exact, postpone, over_period)

def _is_same_period(iDT: dt.datetime, refDT: dt.datetime, unit: str, n: int) -> bool:
    """判断两个时点是否属于同一个周期"""
    if unit == "w":
        # 与 getWeekDateTime/getWeekLastDateTime/getWeekFirstDateTime 的周期判断逻辑一致
        return (iDT.date() - refDT.date()).days == iDT.weekday() - refDT.weekday()
    else:
        return _get_period_key(iDT, unit, n) == _get_period_key(refDT, unit, n)

def _get_period_key(iDT: dt.datetime, unit: str, n: int) -> int:
    """计算时点所属的周期序号（从 0 开始）"""
    if unit == "w":
        return iDT.toordinal() // (7 * n)
    elif unit == "m":
        return iDT.year * 12 + (iDT.month - 1) // n
    elif unit == "q":
        return iDT.year * 4 + (iDT.month - 1) // (3 * n)
    elif unit == "y":
        return iDT.year // n

def _get_target_value(iDT: dt.datetime, unit: str, target: int) -> int:
    """获取时点在指定周期内的目标属性值"""
    if unit == "w":
        return iDT.weekday()  # 0=周一, 6=周日
    elif unit == "m" or unit == "q":
        return iDT.day
    elif unit == "y":
        return iDT.month * 100 + iDT.day

def _transform_by_period(dts: List[dt.datetime], unit: str, n: int,
                          target: Union[int, None], exact: bool, postpone: bool,
                          over_period: bool, take_last: bool = False,
                          take_first: bool = False) -> List[dt.datetime]:
    """按周期分组转换时点序列的通用实现"""
    dts = sorted(dts)

    if target is None and take_last:
        # 取每个周期最后一个时点
        TargetDTs = [dts[0]]
        for iDT in dts:
            if _is_same_period(iDT, TargetDTs[-1], unit, n):
                TargetDTs[-1] = iDT
            else:
                TargetDTs.append(iDT)
        return TargetDTs

    if target is None and take_first:
        # 取每个周期第一个时点
        TargetDTs = [dts[0]]
        for iDT in dts:
            if not _is_same_period(iDT, TargetDTs[-1], unit, n):
                TargetDTs.append(iDT)
        return TargetDTs

    # target 为具体数值（日、星期几等）
    if exact:
        if unit == "y":
            # 对于年频，target 表示月日（如 "0101" 或作为 mmdd 整数）
            if target >= 100:
                target_month = target // 100
                target_day_val = target % 100
                return [iDT for iDT in dts
                        if iDT.month == target_month and iDT.day == target_day_val
                        and _get_period_key(iDT, unit, n) is not None]
            else:
                # target 为一年中的第几天
                return [iDT for iDT in dts
                        if iDT.timetuple().tm_yday == target]
        return [iDT for iDT in dts if _get_target_value(iDT, unit, target) == target]

    if over_period:
        # 构建自然周期目标时点列表，用 searchsorted 二分查找
        dts_arr = np.array(dts, dtype="O")
        DTStrs = [iDT.strftime("%Y%m%d") for iDT in dts_arr]

        if unit == "w":
            NaturalDTStrs = _build_natural_week_targets(dts_arr, n, target, postpone)
        elif unit == "m":
            NaturalDTStrs = _build_natural_month_targets(dts_arr, n, target)
        elif unit == "q":
            NaturalDTStrs = _build_natural_quarter_targets(dts_arr, n, target)
        elif unit == "y":
            NaturalDTStrs = _build_natural_year_targets(dts_arr, n, target)

        if postpone:
            return sorted(set(dts_arr[np.searchsorted(DTStrs, NaturalDTStrs, side="left")]))
        else:
            return sorted(set(dts_arr[np.searchsorted(DTStrs, NaturalDTStrs, side="right") - 1]))

    # over_period=False: 逐周期遍历，在每个周期内找目标时点
    TargetDTs = []
    if postpone:
        # 向后顺延：取每个周期内 >= target 的第一个时点
        for iDT in dts:
            iValue = _get_target_value(iDT, unit, target)
            if iValue >= target:
                if (not TargetDTs) or not _is_same_period(iDT, TargetDTs[-1], unit, n):
                    TargetDTs.append(iDT)
    else:
        # 向前顺延：取每个周期内 <= target 的最后一个时点
        for iDT in dts:
            iValue = _get_target_value(iDT, unit, target)
            if iValue <= target:
                if (not TargetDTs) or not _is_same_period(iDT, TargetDTs[-1], unit, n):
                    TargetDTs.append(iDT)
                else:
                    TargetDTs[-1] = iDT
    return TargetDTs

def _build_natural_week_targets(dts_arr, n, target_weekday, postpone):
    """构建周频自然目标日期字符串列表（over_period=True）

    与 getWeekDateTime 的逻辑保持一致：target_weekday 使用 weekday() 体系（0=周一, 6=周日）
    """
    StartDT = dts_arr[0]
    EndDT = dts_arr[-1]
    tw = target_weekday

    if not postpone:
        StartDT = StartDT + dt.timedelta(tw - StartDT.weekday() + 7 * (StartDT.weekday() > tw))
    else:
        StartDT = StartDT + dt.timedelta(tw - StartDT.weekday())

    StartDT = StartDT + dt.timedelta(0)  # 确保是 datetime 类型
    step = dt.timedelta(7 * n)
    NaturalDTs = getDateTimeSeries(StartDT, EndDT, timedelta=step)
    return [iDT.strftime("%Y%m%d") for iDT in NaturalDTs]

def _build_natural_month_targets(dts_arr, n, target_day):
    """构建月频自然目标日期字符串列表（over_period=True）"""
    StartYear, StartMonth = dts_arr[0].year, dts_arr[0].month
    EndYear, EndMonth = dts_arr[-1].year, dts_arr[-1].month
    nMonth = (EndYear - StartYear) * 12 + EndMonth - StartMonth + 1
    NaturalDTStrs = []
    for i in range(0, nMonth, n):
        iYearNum, iMonthNum = i // 12, i % 12
        iTargetYear = StartYear + iYearNum
        iTargetMonth = StartMonth + iMonthNum
        if iTargetMonth > 12:
            iTargetYear += 1
            iTargetMonth -= 12
        try:
            iDate = dt.date(iTargetYear, iTargetMonth, target_day)
        except ValueError:
            # 目标日超出当月天数（如 2月30日），取当月最后一天
            if iTargetMonth == 12:
                iDate = dt.date(iTargetYear, iTargetMonth, 31)
            else:
                iDate = dt.date(iTargetYear, iTargetMonth + 1, 1) - dt.timedelta(1)
        NaturalDTStrs.append(iDate.strftime("%Y%m%d"))
    return NaturalDTStrs

def _build_natural_quarter_targets(dts_arr, n, target_day):
    """构建季频自然目标日期字符串列表（over_period=True）"""
    StartYear, StartMonth = dts_arr[0].year, dts_arr[0].month
    EndYear, EndMonth = dts_arr[-1].year, dts_arr[-1].month
    # 季度起始月份
    StartQM = ((StartMonth - 1) // (3 * n)) * (3 * n) + 1
    EndQM = ((EndMonth - 1) // (3 * n)) * (3 * n) + 1
    NaturalDTStrs = []
    iYear, iMonth = StartYear, StartQM
    while (iYear < EndYear) or (iYear == EndYear and iMonth <= EndQM):
        try:
            iDate = dt.date(iYear, iMonth, target_day)
        except ValueError:
            if iMonth + 3 * n > 12:
                iDate = dt.date(iYear, 12, 31)
            else:
                iDate = dt.date(iYear, iMonth + 3 * n, 1) - dt.timedelta(1)
        NaturalDTStrs.append(iDate.strftime("%Y%m%d"))
        iMonth += 3 * n
        if iMonth > 12:
            iYear += 1
            iMonth -= 12
    return NaturalDTStrs

def _build_natural_year_targets(dts_arr, n, target_day):
    """构建年频自然目标日期字符串列表（over_period=True）"""
    StartYear = dts_arr[0].year
    EndYear = dts_arr[-1].year
    NaturalDTStrs = []
    for iYear in range(StartYear, EndYear + 1, n):
        if target_day >= 100:
            # mmdd 格式
            target_month = target_day // 100
            target_day_val = target_day % 100
        else:
            # 一年中的第几天
            target_month = 1
            target_day_val = target_day
        try:
            iDate = dt.date(iYear, target_month, target_day_val)
        except ValueError:
            iDate = dt.date(iYear, 12, 31)
        NaturalDTStrs.append(iDate.strftime("%Y%m%d"))
    return NaturalDTStrs
# endregion

# 回溯时点
def _monthrange(year: int, month: int) -> int:
    """返回指定年月的天数"""
    if month == 12:
        return (dt.date(year + 1, 1, 1) - dt.date(year, month, 1)).days
    return (dt.date(year, month + 1, 1) - dt.date(year, month, 1)).days

def lookbackDateTime(idt:Optional[dt.datetime], lookback:Union[Literal["today", "yesterday", "last_friday", "last_month_end"], str]="today", target_day:Union[Literal["exact", "last", "first"], int]="last") -> dt.datetime:
    """给定时点 idt, 返回按照 lookback 和 target_day 规则回溯的时点

    Args:
        idt: 给定的起始时点
        lookback: 回溯期, 预定义值："today", "yesterday", "last_friday", "last_month_end"；
            或用数字+单位表示，比如："1d", "2w", "1m", "3q", "4y"。
            预定义值和天("d")单位会忽略 target_day 参数。
        target_day: 回溯后对具体的时点如何选择，比如 idt:2025-03-05, lookback:"1m",
            * target_day="exact" 表示取 2025-02-05，如果该天不存在，则向前找第一个存在的日子
            * target_day="first" 表示取 2025-02-01
            * target_day="last" 表示取 2025-02-28
            * target_day: int 表示取 2025-02-{target_day}，如果该天不存在，则向前找第一个存在的日子

    Returns:
        回溯后的时点
    """
    iDate = idt.date()
    iTime = idt.time()
    if lookback == "today":
        return dt.datetime.combine(iDate, iTime)
    elif lookback == "yesterday":
        return dt.datetime.combine(iDate - dt.timedelta(days=1), iTime)
    elif lookback == "last_friday":
        days_since_fri = (iDate.weekday() - 4) % 7
        return dt.datetime.combine(iDate - dt.timedelta(days=days_since_fri or 7), iTime)
    elif lookback == "last_month_end":
        first_of_month = iDate.replace(day=1)
        return dt.datetime.combine(first_of_month - dt.timedelta(days=1), iTime)
    import re
    m = re.fullmatch(r"(\d+)([a-zA-Z]+)", lookback)
    if not m:
        raise ValueError(f"无法解析 lookback: {lookback}")
    num = int(m.group(1))
    unit = m.group(2).lower()
    if unit in ("d", "day", "days"):
        iDate -= dt.timedelta(days=num)
    elif unit in ("w", "week", "weeks"):
        iDate -= dt.timedelta(weeks=num)
        if target_day == "first":
            iDate -= dt.timedelta(days=iDate.weekday())
        elif target_day == "last":
            iDate += dt.timedelta(days=6 - iDate.weekday())
        elif target_day != "exact" and not isinstance(target_day, int):
            raise ValueError(f"无效的 target_day: {target_day}")
    elif unit in ("m", "month", "months") or unit in ("q", "quarter", "quarters") or unit in ("y", "year", "years"):
        if unit in ("m", "month", "months"):
            delta_months = num
        elif unit in ("q", "quarter", "quarters"):
            delta_months = num * 3
        else:
            delta_months = num * 12
        iYear, iMonth = idt.year, idt.month
        iYear -= delta_months // 12
        iMonth -= delta_months % 12
        if iMonth < 1:
            iYear -= 1
            iMonth += 12
        if unit in ("q", "quarter", "quarters"):
            if target_day == "first":
                iMonth = ((iMonth - 1) // 3) * 3 + 1
                iDay = 1
            elif target_day == "last":
                iMonth = ((iMonth - 1) // 3) * 3 + 3
                iDay = _monthrange(iYear, iMonth)
            elif target_day == "exact":
                iDay = min(idt.day, _monthrange(iYear, iMonth))
            elif isinstance(target_day, int):
                iDay = min(target_day, _monthrange(iYear, iMonth))
            else:
                raise ValueError(f"无效的 target_day: {target_day}")
        elif unit in ("y", "year", "years"):
            if target_day == "first":
                iMonth = 1
                iDay = 1
            elif target_day == "last":
                iMonth = 12
                iDay = 31
            elif target_day == "exact":
                iDay = min(idt.day, _monthrange(iYear, iMonth))
            elif isinstance(target_day, int):
                iDay = min(target_day, _monthrange(iYear, iMonth))
            else:
                raise ValueError(f"无效的 target_day: {target_day}")
        else:
            iMaxDay = _monthrange(iYear, iMonth)
            if target_day == "first":
                iDay = 1
            elif target_day == "last":
                iDay = iMaxDay
            elif target_day == "exact":
                iDay = min(idt.day, iMaxDay)
            elif isinstance(target_day, int):
                iDay = min(target_day, iMaxDay)
            else:
                raise ValueError(f"无效的 target_day: {target_day}")
        iDate = dt.date(iYear, iMonth, iDay)
    else:
        raise ValueError(f"不支持的 lookback 单位: {unit}")
    return dt.datetime.combine(iDate, iTime)


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