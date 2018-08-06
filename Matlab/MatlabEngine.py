# coding=utf-8
import os
from multiprocessing import Lock

from QuantStudio.QSEnvironment import QSObject,QSArgs,QSError

class MatlabEngine(QSObject):
    """Matlab Engine for Python"""
    def __init__(self, qs_env, sys_args={}):
        self.QSEnv = qs_env
        super().__init__(sys_args)
        self._Engine = None
        self._EngineLock = Lock()
    def __QS_genSysArgs__(self, args=None, **kwargs):
        if args is not None:
            return args
        SysArgs = {"Engine":"新建", 
                   "ExclusiveMode":False, 
                   "Option":"-desktop", 
                   "Async":False,
                   "_ConfigFilePath":self.QSEnv.SysArgs["LibPath"]+os.sep+"MatlabConfig.json"}
        ArgInfo = {}
        ArgInfo["Engine"] = {"type":"SingleOption", "order":0, "range":["新建"]}
        ArgInfo["ExclusiveMode"] = {"type":"Bool", "order":1}
        ArgInfo["Option"] = {"type":"String", "order":2}
        ArgInfo["Async"] = {"type":"Bool", "order":3}
        ArgInfo["_ConfigFilePath"] = {"type":"Path", "order":4, "readonly":True, "operation":"Open", "filter":"Configure files (*.json)"}
        return QSArgs(SysArgs, ArgInfo, self.__QS_onSysArgChanged__)
    def __getstate__(self):
        # Remove the unpicklable entries.
        state = self.__dict__.copy()
        EngineName = self.checkEngine()
        if EngineName is not None:
            state["_Engine"] = {"name":EngineName,"exclusive_mode":(not self._Engine.matlab.engine.isEngineShared())}
        else:
            state["_Engine"] = None
    def __setstate__(self, state):
        if state["_Engine"] is not None:
            self._Engine = None
            if state["_Engine"]["exclusive_mode"]:
                self._EngineLock = Lock()
                self.connect(engine_name=None,exclusive_mode=True)
            else:
                self.connect(engine_name=state["_Engine"]["name"],exclusive_mode=False)
    # 是否链接到引擎, 如果是返回引擎的名字, 否则返回 None
    def checkEngine(self):
        if (self._Engine is not None) and (self._Engine._check_matlab()):# 当前已经链接到一个引擎
            return self._Engine.matlab.engine.engineName()
        else:
            return None
    # 搜索所有可用的 Engine 名称列表
    def findEngine(self):
        try:
            import matlab.engine
        except:
            raise Exception("没有安装好 Matlab Engine for Python!")
        return matlab.engine.find_matlab()
    # 链接至引擎, 如果 engine_name 是 None, 则启动一个新的 engine, 否则链接到给定的 engine
    def connect(self, engine_name=None, exclusive_mode=False, option='-nodesktop'):
        EngineName = self.checkEngine()
        try:
            import matlab.engine
        except:
            raise Exception("没有安装好 Matlab Engine for Python!")
        if EngineName is not None:# 当前已经链接到一个引擎
            if (engine_name is None) or (engine_name==EngineName):
                return 0
            else:
                self._Engine.quit()
        elif engine_name is None:
            self._Engine = matlab.engine.start_matlab(option=option)
            self._Engine.cd(self.QSEnv.SysArgs['MainPath']+os.sep+"Matlab")
            if not exclusive_mode:
                self._Engine.matlab.engine.shareEngine(nargout=0)
            return 0
        AllMatlab = matlab.engine.find_matlab()
        if engine_name in AllMatlab:
            self._Engine = matlab.engine.connect_matlab(name=engine_name)
            self._Engine.cd(self.QSEnv.SysArgs['MainPath']+os.sep+"Matlab")
            return 0
        else:
            raise Exception("MATLAB Engine: %s 不存在!" % engine_name)
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