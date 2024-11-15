# coding=utf-8
import time
import webbrowser

import numpy as np
import pandas as pd
from lxml import etree
from traits.api import Enum

from QuantStudio import __QS_Object__, __QS_Error__
from QuantStudio.FactorDataBase import __QS_BatchContext__
from QuantStudio.FactorDataBase.FactorDB import BatchContext
from QuantStudio.Tools.DataTypeFun import traverseNestedDict, setNestedDictValue, getNestedDictValue

class BaseModule(__QS_Object__):
    """回测模块"""
    def __init__(self, name, sys_args={}, config_file=None, **kwargs):
        self.Name = name
        self._Model = None
        self._Output = {}
        self._QS_isMulti = False# 是否为多重模块
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    
    @property
    def Model(self):
        return self._Model
    
    # 测试开始前的初始化函数
    # 返回: [(factor, ids)]
    def __QS_start__(self, mdl, dts, **kwargs):
        self._Model = mdl
        return []
    
    # 测试结束后的整理函数
    def __QS_end__(self, factor_data):
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


class MultiModule(BaseModule):
    """多模块对比"""
    class __QS_ArgClass__(BaseModule.__QS_ArgClass__):
        CompareCommon = Enum(True, False, arg_type="Bool", label="只对比共同结果", order=-1)
    
    def __init__(self, name="多模块对比", sys_args={}, **kwargs):
        super().__init__(name=name, sys_args=sys_args, **kwargs)
        self._QS_isMulti = True
        self.Modules = []
    
    def output(self, recalculate=False):
        if (not recalculate)  and self._Output: return self._Output
        Output = {"对比结果": {}}
        for i, iModule in enumerate(self.Modules):
            iOutput = iModule.output(recalculate=recalculate)
            iName = str(i)+"-"+iModule.Name
            Output[iName] = iOutput
            for jKeyList, jDF in traverseNestedDict(iOutput):
                for kCol in jDF.columns:
                    kVal = getNestedDictValue(Output["对比结果"], jKeyList+[str(kCol)])
                    if kVal is None:
                        if (i==0) or (not self._QSArgs.CompareCommon):
                            kVal = pd.DataFrame({iName: jDF[kCol]})
                        else:
                            continue
                    else:
                        kVal[iName] = jDF[kCol]
                    setNestedDictValue(Output["对比结果"], jKeyList+[str(kCol)], kVal)
        if self._QSArgs.CompareCommon:
            iOutput = {}
            for jKeyList, jDF in traverseNestedDict(Output["对比结果"]):
                if jDF.shape[1]==len(self.Modules):
                    setNestedDictValue(iOutput, jKeyList, jDF)
            Output["对比结果"] = iOutput
        if not Output["对比结果"]: Output.pop("对比结果")
        self._Output = Output
        return self._Output


class BackTestModel(__QS_Object__):
    """回测模型"""
    def __init__(self, sys_args={}, config_file=None, **kwargs):
        self.Modules = []# 已经添加的测试模块, [测试模块对象]
        self._TestModules = []# 穿透多重模块后得到的所有基本模块列表
        self._Output = {}# 生成的结果集
        self._BatchContext = None
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)
    
    @property
    def BatchContext(self):
        if self._BatchContext is not None:
            return self._BatchContext
        elif __QS_BatchContext__:
            return __QS_BatchContext__[-1]
        return None
    
    @BatchContext.setter
    def BatchContext(self, value):
        if not isinstance(value, BatchContext):
            raise __QS_Error__("BatchContext 必须是批量运算运行时环境对象")
        self._BatchContext = value
    
    # 运行模型
    def _penetrateModule(self, modules):
        AllModules = set()
        for iModule in modules:
            if iModule._QS_isMulti:
                AllModules = AllModules.union(self._penetrateModule(iModule.Modules))
            else:
                AllModules.add(iModule)
        return AllModules
    
    def run(self, dts):
        self._TestModules = list(self._penetrateModule(self.Modules))
        TotalStartT = time.perf_counter()
        print("==========历史回测==========\n1. 初始化")
        Tasks = []
        for j, jModule in enumerate(self._TestModules):
            jTasks = jModule.__QS_start__(mdl=self, dts=dts)
            Tasks += jTasks
        print(("耗时 : %.2f" % (time.perf_counter()-TotalStartT, ))+"\n2. 因子计算")
        StartT = time.perf_counter()
        Context = self.BatchContext
        FactorData = Context.readData(factors=[], ids=[], dts=dts, specific_ids=Tasks, qs_id_key=True)
        print(("耗时 : %.2f" % (time.perf_counter()-StartT, ))+"\n3. 结果生成")
        StartT = time.perf_counter()
        for jModule in self._TestModules: jModule.__QS_end__(factor_data=FactorData)
        print(("耗时 : %.2f" % (time.perf_counter()-StartT, ))+("\n总耗时 : %.2f" % (time.perf_counter()-TotalStartT, ))+"\n"+"="*28)
        self._Output = self.output()
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