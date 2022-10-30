# coding=utf-8
import time
import threading
import webbrowser

import numpy as np
import pandas as pd
from progressbar import ProgressBar
from traitsui.api import Group
from lxml import etree

from QuantStudio import __QS_Object__
from QuantStudio.Tools.AuxiliaryFun import startMultiProcess
from QuantStudio.Tools.QSObjects import QSPipe

class BaseModule(__QS_Object__):
    """回测模块"""
    def __init__(self, name, sys_args={}, config_file=None, **kwargs):
        self.Name = name
        self._Model = None
        self._Output = {}
        self._isStarted = False# 模块是否已经启动
        self._iDT = None# 当前的时点
        self._QS_isMulti = False# 是否为多重模块
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
    # 对象的 HTML 表示
    def _repr_html_(self):
        return ""
    # 生成 HTML 报告的函数, file_path: 给定的文件路径
    def genHTMLReport(self, file_path):
        HTML = self._repr_html_()
        Tree = etree.ElementTree(etree.HTML(HTML))
        Tree.write(file_path)
        return webbrowser.open(file_path)




def _runModel_SingleThread(idx, mdl, end_barrier, start_barrier, ft_lock):
    jModule = mdl._TestModules[idx]
    jFTs = jModule.__QS_start__(mdl=mdl, dts=mdl._QS_TestDateTimes)
    if jFTs is not None:
        jFTs = set(jFTs)
        for jFT in jFTs:
            with ft_lock:
                jFT.start(dts=mdl._QS_TestDateTimes)
    else:
        jFTs = set()
    #print(f"{jModule.Name}: start over!")
    end_barrier.wait()
    for i, iDT in enumerate(mdl._QS_TestDateTimes):
        #print(f"{jModule.Name}: loop {i} begin!")
        start_barrier.wait()
        for jFT in jFTs:
            with ft_lock:
                jFT.move(iDT)
        jModule.__QS_move__(iDT)
        #print(f"{jModule.Name}: loop {i} over!")
        end_barrier.wait()
    #print(f"{jModule.Name}: end begin!")
    start_barrier.wait()
    jModule.__QS_end__()
    for jFT in jFTs:
        with ft_lock:
            jFT.end()
    #print(f"{jModule.Name}: end over!")
    end_barrier.wait()
    return 0

def _runModelProcessThread(args):
    Mdl, Sub2MainQueue, OutputPipe = args.pop("mdl"), args.pop("Sub2MainQueue"), args.pop("OutputPipe")
    nThread = len(args["module_inds"])
    Threads = []
    EndBarrier = threading.Barrier(nThread+1)
    StartBarrier = threading.Barrier(nThread+1)
    FTLock = threading.Lock()
    for jIdx in args["module_inds"]:
        Threads.append(threading.Thread(target=_runModel_SingleThread, args=(jIdx, Mdl, EndBarrier, StartBarrier, FTLock)))
        Threads[-1].start()
    EndBarrier.wait()
    Sub2MainQueue.put(0)
    for i, iDT in enumerate(Mdl._QS_TestDateTimes):
        Mdl._TestDateTimeIndex = i
        Mdl._TestDateIndex.loc[iDT.date()] = i
        EndBarrier.reset()
        StartBarrier.wait()
        StartBarrier.reset()
        EndBarrier.wait()
        Sub2MainQueue.put(1)
    EndBarrier.reset()
    StartBarrier.wait()
    StartBarrier.reset()
    EndBarrier.wait()
    for i in range(nThread):
        Threads[i].join()
    Output = {}
    for j in args["module_inds"]:
        iOutput = Mdl._TestModules[j].output()
        if iOutput: Output[str(j)+"-"+Mdl._TestModules[j].Name] = iOutput
    Sub2MainQueue.put(2)
    OutputPipe.put(Output)
    return 0

def _runModelProcess(args):
    if args["multi_thread"]: return _runModelProcessThread(args)
    Mdl, Sub2MainQueue, OutputPipe = args.pop("mdl"), args.pop("Sub2MainQueue"), args.pop("OutputPipe")
    FTs = set()
    for j in args["module_inds"]:
        jFTs = Mdl._TestModules[j].__QS_start__(mdl=Mdl, dts=Mdl._QS_TestDateTimes)
        if jFTs is not None: FTs.update(set(jFTs))
    for jFT in FTs: jFT.start(dts=Mdl._QS_TestDateTimes)
    Sub2MainQueue.put(0)
    for i, iDT in enumerate(Mdl._QS_TestDateTimes):
        Mdl._TestDateTimeIndex = i
        Mdl._TestDateIndex.loc[iDT.date()] = i
        for jFT in FTs: jFT.move(iDT)
        for j in args["module_inds"]: Mdl._TestModules[j].__QS_move__(iDT)
        Sub2MainQueue.put(1)
    for j in args["module_inds"]: Mdl._TestModules[j].__QS_end__()
    for jFT in FTs: jFT.end()
    Output = {}
    for j in args["module_inds"]:
        iOutput = Mdl._TestModules[j].output()
        if iOutput: Output[str(j)+"-"+Mdl._TestModules[j].Name] = iOutput
    Sub2MainQueue.put(2)
    OutputPipe.put(Output)
    return 0

class BackTestModel(__QS_Object__):
    """回测模型"""
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self.Modules = []# 已经添加的测试模块, [测试模块对象]
        self._QS_TestDateTimes = []# 测试时间点序列, [datetime.datetime]
        self._TestDateTimeIndex = -1# 测试时间点索引
        self._TestDateIndex = pd.Series([], dtype=np.int64)# 测试日期最后一个时间点位于 _QS_TestDateTimes 中的索引
        self._TestModules = []# 穿透多重模块后得到的所有基本模块列表
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
    def _penetrateModule(self, modules):
        AllModules = set()
        for iModule in modules:
            if iModule._QS_isMulti:
                AllModules = AllModules.union(self._penetrateModule(iModule.Modules))
            else:
                AllModules.add(iModule)
        return AllModules
    def run(self, dts, subprocess_num=0, multi_thread=False):
        self._QS_TestDateTimes = sorted(dts)
        self._TestModules = list(self._penetrateModule(self.Modules))
        if subprocess_num>0: return self._runMultiProcs(subprocess_num, multi_thread)
        elif multi_thread: return self._runMultiThread()
        TotalStartT = time.perf_counter()
        print("==========历史回测==========\n1. 初始化")
        FTs = set()
        for jModule in self._TestModules:
            jFTs = jModule.__QS_start__(mdl=self, dts=self._QS_TestDateTimes)
            if jFTs is not None: FTs.update(set(jFTs))
        for jFT in FTs: jFT.start(dts=self._QS_TestDateTimes)
        print(("耗时 : %.2f" % (time.perf_counter()-TotalStartT, ))+"\n2. 循环计算")
        StartT = time.perf_counter()
        with ProgressBar(max_value=len(self._QS_TestDateTimes)) as ProgBar:
            for i, iDT in enumerate(self._QS_TestDateTimes):
                self._TestDateTimeIndex = i
                self._TestDateIndex.loc[iDT.date()] = i
                for jFT in FTs: jFT.move(iDT)
                for jModule in self._TestModules: jModule.__QS_move__(iDT)
                ProgBar.update(i+1)
        print(("耗时 : %.2f" % (time.perf_counter()-StartT, ))+"\n3. 结果生成")
        StartT = time.perf_counter()
        for jModule in self._TestModules: jModule.__QS_end__()
        for jFT in FTs: jFT.end()
        print(("耗时 : %.2f" % (time.perf_counter()-StartT, ))+("\n总耗时 : %.2f" % (time.perf_counter()-TotalStartT, ))+"\n"+"="*28)
        self._Output = self.output()
        return 0
    def _runMultiThread(self):
        nThread = len(self._TestModules)
        Threads = []
        EndBarrier = threading.Barrier(nThread+1)
        StartBarrier = threading.Barrier(nThread+1)
        FTLock = threading.Lock()
        TotalStartT = time.perf_counter()
        print("==========历史回测==========\n1. 初始化")
        for j in range(nThread):
            Threads.append(threading.Thread(target=_runModel_SingleThread, args=(j, self, EndBarrier, StartBarrier, FTLock)))
            Threads[-1].start()
        #FTs = set()
        #for jModule in self.Modules:
            #jFTs = jModule.__QS_start__(mdl=self, dts=self._QS_TestDateTimes)
            #if jFTs is not None: FTs.update(set(jFTs))
        #for jFT in FTs: jFT.start(dts=self._QS_TestDateTimes)
        EndBarrier.wait()
        print(("耗时 : %.2f" % (time.perf_counter()-TotalStartT, ))+"\n2. 循环计算")
        StartT = time.perf_counter()
        with ProgressBar(max_value=len(self._QS_TestDateTimes)) as ProgBar:
            for i, iDT in enumerate(self._QS_TestDateTimes):
                self._TestDateTimeIndex = i
                self._TestDateIndex.loc[iDT.date()] = i
                #for jFT in FTs: jFT.move(iDT)
                EndBarrier.reset()
                StartBarrier.wait()
                StartBarrier.reset()
                EndBarrier.wait()
                ProgBar.update(i+1)
        print(("耗时 : %.2f" % (time.perf_counter()-StartT, ))+"\n3. 结果生成")
        StartT = time.perf_counter()
        EndBarrier.reset()
        StartBarrier.wait()
        #for jFT in FTs: jFT.end()
        StartBarrier.reset()
        EndBarrier.wait()
        for i in range(nThread):
            Threads[i].join()
        print(("耗时 : %.2f" % (time.perf_counter()-StartT, ))+("\n总耗时 : %.2f" % (time.perf_counter()-TotalStartT, ))+"\n"+"="*28)
        self._Output = self.output()
        return 0
    def _runMultiProcs(self, subprocess_num, multi_thread):
        nPrcs = min(subprocess_num, len(self._TestModules))
        Args = {"mdl":self, "module_inds":np.arange(len(self._TestModules)).tolist(), "multi_thread": multi_thread, "OutputPipe":QSPipe()}
        TotalStartT = time.perf_counter()
        print("==========历史回测==========\n1. 初始化")
        Procs, Main2SubQueue, Sub2MainQueue = startMultiProcess(pid="0", n_prc=nPrcs, target_fun=_runModelProcess,
                                                                arg=Args, partition_arg=["module_inds"],
                                                                main2sub_queue=None, sub2main_queue="Single")
        nTask = nPrcs * len(self._QS_TestDateTimes)
        InitStage, CalcStage, EndStage = 0, 0, 0
        self._Output = {}
        while True:
            if CalcStage>=nTask:
                ProgBar.finish()
                print(("耗时 : %.2f" % (time.perf_counter()-CalcStageStartT, ))+"\n3. 结果生成")
                CalcStage = -1
                EndStageStartT = time.perf_counter()
            if EndStage>=nPrcs:
                print(("耗时 : %.2f" % (time.perf_counter()-EndStageStartT, ))+("\n总耗时 : %.2f" % (time.perf_counter()-TotalStartT, ))+"\n"+"="*28)
                break
            iStage = Sub2MainQueue.get()
            if iStage==0:# 初始化阶段
                if InitStage==0:
                    print(("耗时 : %.2f" % (time.perf_counter()-TotalStartT, ))+"\n2. 循环计算")
                    CalcStageStartT = time.perf_counter()
                    ProgBar = ProgressBar(max_value=nTask)
                    ProgBar.start()
                InitStage += 1
            elif iStage==1:# 计算阶段
                CalcStage += 1
                ProgBar.update(CalcStage)
            elif iStage==2:# 结果生成阶段
                EndStage += 1
                iOutput = Args["OutputPipe"].get()
                self._Output.update(iOutput)
                for ijKey, ijOutput in iOutput.items():
                    ijModuleIdx = int(ijKey.split("-")[0])
                    self._TestModules[ijModuleIdx]._Output = ijOutput
        for iPID, iPrcs in Procs.items(): iPrcs.join()
        return 0
    # 计算并输出测试的结果集
    def output(self, recalculate=False):
        self._Output = {}
        for j, jModule in enumerate(self.Modules):
            iOutput = jModule.output(recalculate=recalculate)
            if iOutput: self._Output[str(j)+"-"+jModule.Name] = iOutput
        return self._Output
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