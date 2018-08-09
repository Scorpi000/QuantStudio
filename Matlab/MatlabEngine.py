# coding=utf-8
import os
from multiprocessing import Lock

from traits.api import Enum, Bool, Str

from QuantStudio import __QS_Object__, __QS_MainPath__, __QS_Error__

class MatlabEngine(__QS_Object__):
    """Matlab Engine for Python"""
    ExclusiveMode = Bool(False, arg_type="Bool", label="ExclusiveMode", order=1)
    Option = Str("-desktop", arg_type="String", label="Option", order=2)
    Async = Bool(False, arg_type="Bool", label="Async", order=3)
    def __init__(self, sys_args={}, **kwargs):
        super().__init__(sys_args=sys_args, **kwargs)
        self._Engine = None
        self._EngineLock = Lock()
    def __getstate__(self):
        # Remove the unpicklable entries.
        state = self.__dict__.copy()
        state["_Engine"] = self.EngineName
    def __setstate__(self, state):
        self._Engine = None
        if state["_Engine"] is not None:
            if self.ExclusiveMode:
                self.connect(engine_name=None)
                self._EngineLock = Lock()
            else:
                self.connect(engine_name=state["_Engine"])
    @property
    def EngineName(self):
        if (self._Engine is not None) and (self._Engine._check_matlab()):# 当前已经链接到一个引擎
            return self._Engine.matlab.engine.engineName()
        return None
    # 搜索所有可用的 Engine 名称列表
    def findEngine(self):
        try:
            import matlab.engine
        except:
            raise Exception("没有安装 Matlab Engine for Python!")
        return matlab.engine.find_matlab()
    # 链接至引擎, 如果 engine_name 是 None, 则启动一个新的 engine, 否则链接到给定的 engine
    def connect(self, engine_name=None):
        ConnectedEngineName = self.EngineName
        try:
            import matlab.engine
        except:
            raise __QS_Error__("没有安装 Matlab Engine for Python!")
        if ConnectedEngineName is not None:# 当前已经链接到一个引擎
            if (engine_name is None) or (engine_name==ConnectedEngineName): return 0
            self._Engine.quit()
        elif engine_name is None:
            self._Engine = matlab.engine.start_matlab(option=self.Option)
            self._Engine.cd(__QS_MainPath__+os.sep+"Matlab")
            if not self.ExclusiveMode: self._Engine.matlab.engine.shareEngine(nargout=0)
            return 0
        AllMatlab = matlab.engine.find_matlab()
        if engine_name in AllMatlab:
            self._Engine = matlab.engine.connect_matlab(name=engine_name)
            self._Engine.cd(__QS_MainPath__+os.sep+"Matlab")
            return 0
        else: raise __QS_Error__("MATLAB Engine: %s 不存在!" % engine_name)
    # 请求引擎的使用权, 返回引擎
    def acquireEngine(self):
        self._EngineLock.acquire()
        return self._Engine
    # 释放引擎的使用权
    def releaseEngine(self):
        self._EngineLock.release()
        return 0
    # 断开 MATLAB 引擎
    def disconnect(self):
        try:
            self._Engine.quit()
        except:
            pass
        finally:
            self._Engine = None
        return 0