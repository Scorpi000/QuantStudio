# coding=utf-8
"""基本的因子运算"""
import datetime as dt

import numpy as np
import pandas as pd

from QuantStudio.FactorDataBase.FactorDB import Factor
from QuantStudio.FactorDataBase.FactorOperation import PointOperator

# ----------------------单点运算--------------------------------
class Neg(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "neg", "入参数": 1, "最大入参数": 1, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return - x[0]


class Abs(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "abs", "入参数": 1, "最大入参数": 1, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return np.abs(x[0])

class Not(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "not", "入参数": 1, "最大入参数": 1, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return ~ x[0].astype(bool)

class Add(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "add", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] + x[1]


class Sub(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "sub", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] - x[1]


class Mul(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "mul", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] * x[1]

class Div(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "div", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] / np.where(x[1]==0, np.nan, x[1])


class FloorDiv(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "floordiv", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] // np.where(x[1]==0, np.nan, x[1])


class Mod(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "mod", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] % np.where(x[1]==0, np.nan, x[1])


class Pow(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "pow", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        r = x[0] ** x[1]
        r[np.isinf(r)] = np.nan
        return r


class And(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "and", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0].astype(bool) & x[1].astype(bool)


class Or(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "or", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0].astype(bool) | x[1].astype(bool)


class Xor(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "xor", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0].astype(bool) ^ x[1].astype(bool)


class LT(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "lt", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] < x[1]


class LE(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "le", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] <= x[1]


class GT(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "gt", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] > x[1]


class GE(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "ge", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] >= x[1]


class Eq(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "eq", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] == x[1]


class Neq(PointOperator):
    class __QS_ArgClass__(PointOperator.__QS_ArgClass__):
        def __QS_initArgValue__(self, args={}):
            Args = {"名称": "neq", "入参数": 2, "最大入参数": 2, "运算时点": "多时点", "运算ID": "多ID"}
            Args.update(args)
            return super().__QS_initArgValue__(args=Args)
    
    def calculate(self, f, idt, iid, x, args):
        return x[0] != x[1]

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
    from QuantStudio.FactorDataBase.FactorDB import DataFactor, Factorize
    
    np.random.seed(0)
    IDs = [f"00000{i}.SZ" for i in range(1, 6)]
    DTs = [dt.datetime(2020, 1, 1) + dt.timedelta(i) for i in range(4)]
    Factor1 = DataFactor(name="Factor1", data=1)
    Factor2 = DataFactor(name="Factor2", data=pd.DataFrame(np.random.randn(len(DTs), len(IDs)), index=DTs, columns=IDs))
    
    # 表达式方式
    Factor3 = Factorize(Factor1 - Factor2, factor_name="Factor3")
    
    Factor4 = sub(Factor1, Factor2, factor_name="Factor4")
    
    Factor5 = Sub()(Factor1, Factor2, factor_name="Factor5")
    
    Factor6 = Factorize(1 - Factor2, factor_name="Factor6")
    
    print(Factor1.readData(ids=IDs, dts=DTs))
    print(Factor2.readData(ids=IDs, dts=DTs))
    print(Factor3.readData(ids=IDs, dts=DTs))
    print(Factor4.readData(ids=IDs, dts=DTs))
    print(Factor5.readData(ids=IDs, dts=DTs))
    print(Factor6.readData(ids=IDs, dts=DTs))
    
    print("===")    
