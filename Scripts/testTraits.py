# coding=utf-8
import datetime as dt
#from traits.etsconfig.api import ETSConfig
#ETSConfig.toolkit = 'qt4'
from traits.api import HasTraits, Str, Instance, Button, Date, Time, Int, Enum, List, Range, Any, HasPrivateTraits, Float, ListStr, DictStrStr, on_trait_change, Dict, Date, Directory, File, Password, Event
from traitsui.menu import OKButton, CancelButton
from traitsui.api import Item, View, Group, CheckListEditor, ListEditor, ListStrEditor, SetEditor, ValueEditor, CSVListEditor
from QuantStudio import __QS_Object__


class QSArgs(HasTraits):
    SingleOption = Enum(None)
    MultiOption = ListStr()
    ArgDict = Dict(value={"aha":"d1"}, key_trait=Str, value_trait=Enum("d1", "d2"))
    Obj = Instance("_TradeLimit", allow_none=False, arg_type="ArgObject", label="买入限制", order=2)
    ChangeEvent = Event()
    @on_trait_change("SingleOption")
    def _on_SingleOption_changed(self, obj, name, old, new):
        print(old+"-->"+new)
    @on_trait_change("MultiOption[]")
    def _on_MultiOption_changed(self, obj, name, old, new):
        print(str(old)+"-->"+str(new))
    @on_trait_change("ArgDict[]")
    def _on_ArgDict_changed(self):
        print(str(self.ArgDict))
    #@on_trait_change("Obj.aha")
    #def _on_Obj_changed(self):
        #print(str(self.Obj.aha))
    #@on_trait_change("Obj")
    #def _on_Obj_changed2(self, obj, name, old, new):
        #print(str(old)+"-->"+str(new))
    #def _ChangeEvent_fired(self, new):
        #print(new)


class _TradeLimit(__QS_Object__):
    """交易限制"""
    LimitIDFilter = Str(arg_type="IDFilter", label="禁止条件", order=0)
    TradeFee = Float(0.003, arg_type="Double", label="交易费率", order=1)
    MinUnit = Int(0, arg_type="Integer", label="最小单位", order=2)
    MarketOrderVolumeLimit = Float(0.1, arg_type="Double", label="市价单成交量限比", order=3)
    LimitOrderVolumeLimit = Float(0.1, arg_type="Double", label="限价单成交量限比", order=4)
    #view = View(Item("LimitIDFilter"), buttons=[OKButton, CancelButton], resizable=True, title="设置参数")
    def __init__(self, direction, sys_args={}, **kwargs):
        self._Direction = direction
        super().__init__(**kwargs)
        self.trait_view(name="QSView", view_element=View(*self.getViewItems()[0], buttons=[OKButton, CancelButton], resizable=True, title=getattr(self, "Name", "设置参数")))

class QSObject(__QS_Object__):
    Limit = Instance(_TradeLimit, arg_type="ArgObject", label="条件", order=0)

#TestObject = QSObject()
#TestObject.Limit = _TradeLimit("Buy")
#TestObject.setArgs()

class testC(HasTraits):
    aha = Int(0, arg_type="Integer", label="最小单位", order=2)
    pass

class testEvent(HasTraits):
    def _ChangeEvent_fired(self, new):
        print(new)
#Args = QSArgs()
#Args.Obj = _TradeLimit("buy")

#view = View(Item("QSArgs.MultiOption", editor=SetEditor(name="MultiOption")),
            #Item("QSArgs.SingleOption"),
            #Item("QSArgs.ArgDict", editor=ValueEditor()),
            #Item("QSArgs.Obj"),
            #buttons=[OKButton, CancelButton],
            #resizable=True,
            #title="IC")
#Rslt = Args.configure_traits(view=view, context={"QSArgs":Args})

class DateTimeList(__QS_Object__):
    #StartDate = Date(arg_type="Date", label="起始日期", order=0)
    #EndDate = Date(arg_type="Date", label="结束日期", order=1)
    #StartTime = Time(arg_type="Date", label="起始时间", order=2)
    #EndTime = Time(arg_type="Date", label="结束时间", order=3)
    #Exchange = Enum("SSE", "SZSE", "CFFEE", arg_type="SingleOption", label="交易所", order=4)
    #DateType = Enum(None, "交易日", "自然日", arg_type="SingleOption", label="日期类型", order=5)
    #DateFreq = Enum(None, "月底日", "周末日", arg_type="SingleOption", label="日期频率", order=6)
    #DateList = List(Date, arg_type="SingleOption", label="日期序列", order=7)
    #TimeList = List(Time, arg_type="SingleOption", label="时间序列", order=8)
    Obj = List(label="设置", arg_type="DateList", order=9)
    
Args = DateTimeList()
Rslt = Args.setArgs()

#Args.SingleOption = "a2"
#print(Args.SingleOption)
#Args.MultiOption.append("aha")
#Args.MultiOption = []
#Args.ArgDict["aha"] = "d1"
#Args.ArgDict = {}
#Args.Obj = testC()
#Args.Obj.aha = 1
#Args.Obj.aha1 = 2
#Args.Obj = testC()
#EventListener = testEvent()
#Args.on_trait_event(handler=EventListener._ChangeEvent_fired, name="ChangeEvent")
#Args.add_trait_listener(EventListener, prefix="ChangeEvent")
#Args.remove_trait_listener(EventListener)
#Args.ChangeEvent = 1
print("="*28)
pass