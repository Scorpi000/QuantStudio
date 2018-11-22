# coding=utf-8
import os
import time
import shutil
import datetime as dt
import webbrowser

import numpy as np
import pandas as pd
from progressbar import ProgressBar
from traits.api import List, Instance, Str
from traitsui.api import View, Item, Group
from traitsui.menu import OKButton, CancelButton
from lxml import etree

from QuantStudio import __QS_Error__, __QS_Object__, __QS_MainPath__
from QuantStudio.FactorDataBase.FactorDB import FactorDB

class BaseModule(__QS_Object__):
    """回测模块"""
    Name = Str("回测模块")
    def __init__(self, name, sys_args={}, config_file=None, **kwargs):
        self.Name = name
        self._Model = None
        self._Output = {}
        self._isStarted = False# 模块是否已经启动
        self._iDT = None# 当前的时点
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    @property
    def Model(self):
        return self._Model
    # 测试开始前的初始化函数
    def __QS_start__(self, mdl, dts, **kwargs):
        self._Model = mdl
        self._isStarted = True
        return ()
    # 测试至某个时点的计算函数
    def __QS_move__(self, idt, **kwargs):
        self._iDT = idt
        return 0
    # 测试结束后的整理函数
    def __QS_end__(self):
        self._isStarted = False
        self._iDT = None
        return 0
    # 计算并输出测试的结果集
    def output(self, recalculate=False):
        return self._Output
    # 生成 Excel 报告的函数, xl_book: 给定的 Excel 工作簿对象, sheet_name: 给定的工作表名
    def genExcelReport(self, xl_book, sheet_name):
        return 0
    # 对象的 HTML 表示
    def _repr_html_(self):
        return ""
    # 生成 HTML 报告的函数, file_path: 给定的文件路径
    def genHTMLReport(self, file_path):
        HTML = self._repr_html_()
        Tree = etree.ElementTree(etree.HTML(HTML))
        Tree.write(file_path)
        return webbrowser.open(file_path)


class BackTestModel(__QS_Object__):
    """回测模型"""
    Modules = List(BaseModule)# 已经添加的测试模块, [测试模块对象]
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self._QS_TestDateTimes = []# 测试时间点序列, [datetime.datetime]
        self._TestDateTimeIndex = -1# 测试时间点索引
        self._TestDateIndex = pd.Series([], dtype=np.int64)# 测试日期最后一个时间点位于 _QS_TestDateTimes 中的索引
        self._Output = {}# 生成的结果集
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    # 当前时点, datetime.datetime
    @property
    def DateTime(self):
        return self._QS_TestDateTimes[self._TestDateTimeIndex]
    # 当前时间点在整个回测时间序列中的位置索引, int
    @property
    def DateTimeIndex(self):
        return self._TestDateTimeIndex
    # 截止当前的时间点序列, [datetime.datetime]
    @property
    def DateTimeSeries(self):
        return self._QS_TestDateTimes[:self._TestDateTimeIndex+1]
    # 截止到当前日期序列在时间点序列中的索引, Series(int, index=[日期])
    @property
    def DateIndexSeries(self):
        return self._TestDateIndex
    def getViewItems(self, context_name=""):
        Prefix = (context_name+"." if context_name else "")
        Groups, Context = [], {}
        for j, jModule in enumerate(self.Modules):
            jItems, jContext = jModule.getViewItems(context_name=Prefix+"Module"+str(j))
            Groups.append(Group(*jItems, label=str(j)+"-"+jModule.Name))
            Context.update(jContext)
            Context[Prefix+"Module"+str(j)] = jModule
        return (Groups, Context)
    # 运行模型
    def run(self, dts):
        self._QS_TestDateTimes = sorted(dts)
        TotalStartT = time.clock()
        print("==========历史回测==========", "1. 初始化", sep="\n", end="\n")
        FactorDBs = set()
        for jModule in self.Modules:
            jDBs = jModule.__QS_start__(mdl=self, dts=dts)
            if jDBs is not None: FactorDBs.update(set(jDBs))
        for jDB in FactorDBs: jDB.start(dts=dts)
        print(('耗时 : %.2f' % (time.clock()-TotalStartT, )), "2. 循环计算", sep="\n", end="\n")
        StartT = time.clock()
        with ProgressBar(max_value=len(self._QS_TestDateTimes)) as ProgBar:
            for i, iDT in enumerate(self._QS_TestDateTimes):
                self._TestDateTimeIndex = i
                self._TestDateIndex.loc[iDT.date()] = i
                for jDB in FactorDBs: jDB.move(iDT)
                for jModule in self.Modules: jModule.__QS_move__(iDT)
                ProgBar.update(i+1)
        print(('耗时 : %.2f' % (time.clock()-StartT, )), "3. 结果生成", sep="\n", end="\n")
        StartT = time.clock()
        for jModule in self.Modules: jModule.__QS_end__()
        for jDB in FactorDBs: jDB.end()
        print(('耗时 : %.2f' % (time.clock()-StartT, )), ("总耗时 : %.2f" % (time.clock()-TotalStartT, )), "="*28, sep="\n", end="\n")
        self._Output = self.output(recalculate=True)
        return 0
    # 计算并输出测试的结果集
    def output(self, recalculate=False):
        if not recalculate: return self._Output
        self._Output = {}
        for j, jModule in enumerate(self.Modules):
            iOutput = jModule.output(recalculate=True)
            if iOutput: self._Output[str(j)+"-"+jModule.Name] = iOutput
        return self._Output
    # 生成 Excel 报告
    def genExcelReport(self, file_path):
        shutil.copy(__QS_MainPath__+os.sep+"Resource"+os.sep+"SimpleSectionFactorTemplate.xlsx", file_path)
        xlBook = xw.Book(file_path)
        NewSheet = xlBook.sheets.add(name="占位表")
        for i, iModule in enumerate(self.Modules): iModule.genExcelReport(xlBook, str(i)+"-"+iModule.Name)
        xlBook.app.display_alerts = False
        xlBook.sheets["IC"].delete()
        xlBook.sheets["因子换手率"].delete()
        xlBook.sheets["因子值行业分布"].delete()
        xlBook.sheets["分位数组合"].delete()
        xlBook.sheets["IC的衰减"].delete()
        if xlBook.sheets.count>1: xlBook.sheets["占位表"].delete()
        xlBook.app.display_alerts = True
        xlBook.save()
        xlBook.app.quit()
        return 0
    # 对象的 HTML 表示
    def _repr_html_(self):
        HTML = ''
        SepStr = '<HR style="FILTER: alpha(opacity=100,finishopacity=0,style=3)" width="90%" color=#987cb9 SIZE=5><div align="center" style="font-size:1.17em"><strong>{Module}</strong></div>'
        for i, iModule in enumerate(self.Modules): HTML += SepStr.format(Module=str(i)+". "+iModule.Name) + iModule._repr_html_()
        return HTML
    # 生成 HTML 报告
    def genHTMLReport(self, file_path):
        HTML = self._repr_html_()
        Tree = etree.ElementTree(etree.HTML(HTML))
        Tree.write(file_path)
        return webbrowser.open(file_path)