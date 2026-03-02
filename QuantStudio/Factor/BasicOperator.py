# coding=utf-8
"""基本的因子运算"""
import datetime as dt
from typing import Dict

import numpy as np
import pandas as pd

from QuantStudio.Core.Factor import Factor
from QuantStudio.Core.FactorOperation import DerivativeFactor, PointOperator

# ----------------------单点运算--------------------------------
class Rename(PointOperator):
    def __init__(self, args={}, config_file=None, **kwargs):
        Args = {"Name": "rename", "DataType": "object"} | args | {"Arity": 1, "DTMode": "多时点", "IDMode": "多ID"}
        return super().__init__(args=Args, config_file=config_file, **kwargs)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0]
    
    def __call__(self, f: Factor, factor_name: str, factor_args:Dict={}, **kwargs):
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
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        
        def __init__(self, /, **data):
            Args = {"Name": "neg", "Arity": 1, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return - x[0]

    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class Abs(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "abs", "Arity": 1, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return np.abs(x[0])
    
    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class Not(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "not", "Arity": 1, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return ~ x[0].astype(bool)
    
    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class Add(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "add", "Arity": 2, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] + x[1]
    
    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)


class Sub(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "sub", "Arity": 2, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] - x[1]
    
    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)


class Mul(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "mul", "Arity": 2, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] * x[1]
    
    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class Div(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "div", "Arity": 2, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] / np.where(x[1]==0, np.nan, x[1])
    
    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)


class FloorDiv(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "floordiv", "Arity": 2, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] // np.where(x[1]==0, np.nan, x[1])
    
    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)


class Mod(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "mod", "Arity": 2, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] % np.where(x[1]==0, np.nan, x[1])
    
    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)


class Pow(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "pow", "Arity": 2, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        r = x[0] ** x[1]
        r[np.isinf(r)] = np.nan
        return r
    
    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)


class And(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "and", "Arity": 2, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0].astype(bool) & x[1].astype(bool)

    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class Or(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "or", "Arity": 2, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0].astype(bool) | x[1].astype(bool)
    
    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)


class Xor(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "xor", "Arity": 2, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0].astype(bool) ^ x[1].astype(bool)

    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)

class LT(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "lt", "Arity": 2, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] < x[1]
    
    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)


class LE(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "le", "Arity": 2, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] <= x[1]
    
    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)


class GT(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "gt", "Arity": 2, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] > x[1]
    
    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)


class GE(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "ge", "Arity": 2, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] >= x[1]
    
    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)


class Eq(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "eq", "Arity": 2, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] == x[1]
    
    def __call__(self, *x, factor_args:Dict={}, **kwargs):
        factor_args = {"CacheEnabled": False} | factor_args
        return super().__call__(*x, factor_args=factor_args, **kwargs)


class Neq(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __init__(self, /, **data):
            Args = {"Name": "neq", "Arity": 2, "DTMode": "多时点", "IDMode": "多ID"}
            Args.update(data)
            return super().__init__(**Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] != x[1]
    
    def __call__(self, *x, factor_args:Dict={}, **kwargs):
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


if __name__=="__main__":
    from QuantStudio.Core.Factor import DataFactor
    
    np.random.seed(0)
    IDs = [f"00000{i}.SZ" for i in range(1, 6)]
    DTs = [dt.datetime(2020, 1, 1) + dt.timedelta(i) for i in range(4)]
    Factor1 = DataFactor(data=1, args={"Name": "Factor1"})
    Factor2 = DataFactor(data=pd.DataFrame(np.random.randn(len(DTs), len(IDs)), index=DTs, columns=IDs), args={"Name": "Factor2"})
    
    Factor3 = rename(Factor1 + Factor2, factor_name="Factor3")
    Factor4 = sub(Factor1, Factor2, factor_name="Factor4")
    print(Factor4.model_dump())
    print(Factor4.QSID)
    Factor5 = Sub()(Factor1, Factor2, factor_name="Factor5")
    print(Factor5.model_dump())
    print(Factor5.QSID)
    Factor6 = rename(1 - Factor2, factor_name="Factor6")
    
    print(Factor1.readData(ids=IDs, dts=DTs))
    print(Factor2.readData(ids=IDs, dts=DTs))
    print(Factor3.readData(ids=IDs, dts=DTs))
    print(Factor4.readData(ids=IDs, dts=DTs))
    print(Factor5.readData(ids=IDs, dts=DTs))
    print(Factor6.readData(ids=IDs, dts=DTs))
    

    print("===")
