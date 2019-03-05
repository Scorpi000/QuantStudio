# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from queue import PriorityQueue
from collections import defaultdict
from threading import Thread

# 事件对象
@dataclass(order=True)
class Event:
    Type: str = field(default="EVENT", compare=False)# 事件类型, 事件的唯一标识符, QS 系统产生的事件均以 QS_ 开始
    Priority: int = 0# 事件的优先级, 优先级越小的事件越早处理
    Data: Any = field(default=None, compare=False)# 事件的相关数据
    _Ref: int = 0# 事件索引, 用于保证在优先级相同时按照 FIFO 组织, 由事件引擎维护该值

# 事件驱动引擎
class EventEngine(object):
    """
    事件驱动引擎
    事件监听函数必须定义为输入参数仅为一个 event 对象, 即:
    函数
    def func(event)
        ...
    
    对象方法
    def method(self, event)
        ...
    """
    def __init__(self):
        self._EventQueue = PriorityQueue()# 事件队列
        self._MaxEventRef = 0# 当前最大的事件索引
        self._Active = False# 事件引擎开关
        self._Thread = Thread(target=self._run)# 事件处理线程
        self._Handlers = defaultdict(list)# 已经注册的事件处理回调函数
        self._GeneralHandlers = []# 已经注册的通用事件处理回调函数, 对所有事件均调用
    # 运行事件驱动引擎
    def _run(self):
        while self._Active:
            iEvent = self._EventQueue.get(block=True)
            if iEvent.Type in self._Handlers:
                for iHandler in self._Handlers[iEvent.Type]: iHandler(iEvent)
            for iHandler in self._GeneralHandlers: iHandler(iEvent)
            if iEvent.Type=="EXIT":# 退出事件
                self._Active = False
                break
    # 启动引擎
    def start(self):
        self._Active = True
        self._Thread.start()# 启动事件处理线程
    # 停止引擎
    def exit(self):
        self.put(Event(Type="EXIT"))
        self._Thread.join()# 等待事件处理线程退出
    # 等待引擎结束
    def wait(self):
        self._Thread.join()
    # 注册事件处理回调函数, 如果 event_type 是 None 表示注册通用事件处理函数
    def register(self, handler, event_type=None):
        # 若要注册的处理器不在该事件的处理器列表中，则注册该事件
        if event_type is None:
            if handler not in self._GeneralHandlers:
                self._GeneralHandlers.append(handler)
        elif handler not in self._Handlers[event_type]:
            self._Handlers[event_type].append(handler)
    # 注销事件处理回调函数, 如果 event_type 是 None 表示注册通用事件处理函数
    def unregister(self, handler, event_type=None):
        # 如果该函数存在于列表中, 则移除
        if event_type is None:
            if handler in self._GeneralHandlers:
                self._GeneralHandlers.remove(handler)
        elif handler in self._Handlers[event_type]:
            self._Handlers[event_type].remove(handler)
            # 如果函数列表为空，则从引擎中移除该事件类型
            if not self._Handlers[event_type]:
                del self._Handlers[event_type]
    # 向事件队列中存入事件
    def put(self, event, block=True, timeout=None):
        event._Ref = self._MaxEventRef = self._MaxEventRef + 1
        self._EventQueue.put(event, block=block, timeout=timeout)

if __name__ == '__main__':
    # 测试代码
    import sys
    import time
    import datetime as dt
    
    # 运行在定时器线程中的循环函数
    def runTimer(ee):
        TotalTime = 0
        Active = True
        while Active:
            # 创建定时器事件
            if TotalTime>=10:
                iEvent = Event(Type="EXIT")
                Active = False
            else:
                iEvent = Event(Type="EVENT_TIMER")
            # 向队列中存入计时器事件
            ee.put(iEvent)    
            # 等待
            time.sleep(1)
            TotalTime += 1
    
    def TestFun(event):
        print(f"处理定时器事件：{dt.datetime.now()}")
    
    EE = EventEngine()
    EE.register(TestFun, )
    EE.start()
    
    # 创建一个定时器线程
    Timer = Thread(target=runTimer, args=(EE,))
    Timer.start()
    
    EE.wait()
    #from PyQt5.QtCore import QCoreApplication
    #App = QCoreApplication(sys.argv)
    #App.exec_()