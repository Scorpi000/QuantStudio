# coding=utf-8
"""基本的因子运算"""
import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

from QuantStudio.Factor.Factor import Factor
from QuantStudio.Factor.FactorOperation import PointOperator, PointOperation, OperandSpec

# ----------------------单点运算--------------------------------
class Rename(PointOperator):
    """重命名因子"""

    def __init__(self, args:dict={}, config_file:Optional[str]=None, **kwargs):
        Args = {"Name": "rename", "DataType": "object"} | args | {"OperandInfo": (OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0]
    
    def __call__(self, f:Factor, factor_name:str, factor_args:dict={}, **kwargs) -> Factor:
        if not f.FactorTable:
            factor = f.new(args=factor_args | {"Name": factor_name})
            if "logger" in kwargs: factor._QS_Logger = kwargs["logger"]
            return factor
        DataType = f.getMetaData(key="DataType")
        factor_args = {"CacheEnabled": False} | factor_args | {"Name": factor_name}
        if DataType != self._QSArgs.DataType:
            return super(Rename, self.new(args={"DataType": DataType})).__call__(f, factor_args=factor_args, **kwargs)
        else:
            return super().__call__(f, factor_args=factor_args, **kwargs)

class Neg(PointOperator):
    """取负值"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "neg", "OperandInfo": (OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return - x[0]

    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class Abs(PointOperator):
    """绝对值"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "abs", "OperandInfo": (OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return np.abs(x[0])
    
    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class Not(PointOperator):
    """非"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "not", "OperandInfo": (OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return ~ x[0].astype(bool)
    
    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class Add(PointOperator):
    """加法"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "add", "OperandInfo": (OperandSpec(), OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] + x[1]
    
    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class Sub(PointOperator):
    """减法"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "sub", "OperandInfo": (OperandSpec(), OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] - x[1]
    
    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class Mul(PointOperator):
    """乘法"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "mul", "OperandInfo": (OperandSpec(), OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] * x[1]
    
    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class Div(PointOperator):
    """除法"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "div", "OperandInfo": (OperandSpec(), OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] / np.where(x[1]==0, np.nan, x[1])
    
    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class FloorDiv(PointOperator):
    """整除"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "floordiv", "OperandInfo": (OperandSpec(), OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] // np.where(x[1]==0, np.nan, x[1])
    
    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class Mod(PointOperator):
    """取余"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "mod", "OperandInfo": (OperandSpec(), OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] % np.where(x[1]==0, np.nan, x[1])
    
    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class Pow(PointOperator):
    """乘方"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "pow", "OperandInfo": (OperandSpec(), OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        r = x[0] ** x[1]
        r[np.isinf(r)] = np.nan
        return r
    
    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class And(PointOperator):
    """与"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "and", "OperandInfo": (OperandSpec(), OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0].astype(bool) & x[1].astype(bool)

    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class Or(PointOperator):
    """或"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "or", "OperandInfo": (OperandSpec(), OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0].astype(bool) | x[1].astype(bool)
    
    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class Xor(PointOperator):
    """异或"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "xor", "OperandInfo": (OperandSpec(), OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0].astype(bool) ^ x[1].astype(bool)

    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class LT(PointOperator):
    """小于"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "lt", "OperandInfo": (OperandSpec(), OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] < x[1]
    
    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class LE(PointOperator):
    """小于等于"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "le", "OperandInfo": (OperandSpec(), OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] <= x[1]
    
    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class GT(PointOperator):
    """大于"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "gt", "OperandInfo": (OperandSpec(), OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] > x[1]
    
    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class GE(PointOperator):
    """大于等于"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "ge", "OperandInfo": (OperandSpec(), OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] >= x[1]
    
    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class Eq(PointOperator):
    """等于"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "eq", "OperandInfo": (OperandSpec(), OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] == x[1]
    
    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class Neq(PointOperator):
    """不等于"""

    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "neq", "OperandInfo": (OperandSpec(), OperandSpec(),), "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] != x[1]
    
    def __call__(self, *x:Factor, factor_args:dict={}, **kwargs) -> PointOperation:
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

rename = Rename()
neg = Neg()
qs_abs = Abs()
qs_not = Not()
add = Add()
sub = Sub()
mul = Mul()
div = Div()
floordiv = FloorDiv()
mod = Mod()
qs_pow = Pow()
qs_and = And()
qs_or = Or()
xor = Xor()
lt = LT()
le = LE()
gt = GT()
ge = GE()
eq = Eq()
neq = Neq()

