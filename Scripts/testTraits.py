# coding=utf-8
#from traits.etsconfig.api import ETSConfig
#ETSConfig.toolkit = 'qt4'
from traits.api import HasTraits, Str, Instance, Int, Enum, List, Range, Any, HasPrivateTraits, Float, ListStr, DictStrStr, on_trait_change, Dict, Date, Directory, File, Password, Event
from traitsui.menu import OKButton, CancelButton
from traitsui.api import Item, View, Group, CheckListEditor, ListEditor, ListStrEditor, SetEditor, ValueEditor, CSVListEditor


class QSArgs(HasTraits):
    SingleOption = Enum(None)
    MultiOption = ListStr()
    ArgDict = Dict(value={"aha":"d1"}, key_trait=Str, value_trait=Enum("d1", "d2"))
    Obj = Instance("testC", allow_none=False)
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
    @on_trait_change("Obj")
    def _on_Obj_changed2(self, obj, name, old, new):
        print(str(old)+"-->"+str(new))
    #def _ChangeEvent_fired(self, new):
        #print(new)

class testC(HasTraits):
    aha = Int(0)
    pass

class testEvent(HasTraits):
    def _ChangeEvent_fired(self, new):
        print(new)
Args = QSArgs()

view = View(Item(name="MultiOption", editor=SetEditor(name="MultiOption")),
            Item(name="SingleOption"),
            Item(name="ArgDict", editor=ValueEditor()),
            buttons=[OKButton, CancelButton],
            resizable=True,
            title="IC")
Args.configure_traits(view=view)



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