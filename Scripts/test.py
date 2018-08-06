# multi_object_view.py -- Sample code to show multi-object view
#                         with context

from traits.api import HasTraits, Str, Int, Bool
from traitsui.api import View, Group, Item

# Sample class
class House(HasTraits):
   address = Str
   bedrooms = Int
   pool = Bool
   price = Int

# View object designed to display two objects of class 'House'
Items = [Item('h1.address', resizable=True), Item('h1.bedrooms'), Item('h1.pool'), Item('h1.price')]
#for iItem in Items:
   #iItem.name = "h1."+iItem.name
comp_view = View(Group(*Items, label="h1"), title = 'House Comparison')
# A pair of houses to demonstrate the View
house1 = House(address='4743 Dudley Lane',
               bedrooms=3,
               pool=False,
               price=150000)
house2 = House(address='11604 Autumn Ridge',
               bedrooms=3,
               pool=True,
               price=200000)

# ...And the actual display command
house1.configure_traits(view=comp_view, context={'h1':house1,
                                                 'h2':house2})