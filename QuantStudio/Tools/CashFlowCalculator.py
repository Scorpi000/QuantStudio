# coding=utf-8
"""现金流计算"""

import numpy as np

# 说明:
# 对于投资者, 负现金流表示现金流出(投入资金), 正现金流表示现金流入(获得收入)

# 规则现金流, 固定收益率模型

# 计算现值
# numpy.pv(rate, nper, pmt, fv=0, when='end')

# 计算终值
# numpy.fv(rate, nper, pmt, pv, when='end')

# 计算每期现金流
# numpy.pmt(rate, nper, pv, fv=0, when='end')

# 计算每期收益率
# numpy.rate(nper, pmt, pv, fv, when='end', guess=None, tol=None, maxiter=100)

# 计算期数
# numpy.nper(rate, pmt, pv, fv=0, when='end')


# 不规则现金流, 固定收益率模型

# 计算内部收益率 IRR
# numpy.irr(values)

# 计算净现值
# numpy.npv(rate, values)


# 不规则现金流, 动态收益率模型

# 检查并调整输入变量
# rate: array((T, M)), array((T, )), float
# pmt: array((T, M)), array((T, )), float
# pv: array((M, )), float
# fv: array((M, )), float
# when: at the beginning (when = "begin") or the end (when = "end") of each period
def _adjust_input(rate=None, pmt=None, pv=None, fv=None, when="end"):
    InputShapes = np.ones((4, 2), dtype=int)
    ReturnScalar = True
    if rate is not None:
        InputShapes[0, 0:np.ndim(rate)] = np.shape(rate)
        ReturnScalar = (ReturnScalar and (np.ndim(rate)<=1))
    if pmt is not None:
        InputShapes[1, 0:np.ndim(pmt)] = np.shape(pmt)
        ReturnScalar = (ReturnScalar and (np.ndim(pmt)<=1))
    if pv is not None:
        InputShapes[2, 1:np.ndim(pv)+1] = np.shape(pv)
        ReturnScalar = (ReturnScalar and (np.ndim(pv)<1))
    if fv is not None:
        InputShapes[3, 1:np.ndim(fv)+1] = np.shape(fv)
        ReturnScalar = (ReturnScalar and (np.ndim(fv)<1))
    T, M = np.max(InputShapes, axis=0)
    if not np.all(((InputShapes[:, 0]==1) | (InputShapes[:, 0]==T)) & ((InputShapes[:, 1]==1) | (InputShapes[:, 1]==M))):
        raise Exception("There is an input having wrong dimention!")
    if rate is not None:
        rate = np.repeat(np.repeat(np.reshape(rate, InputShapes[0,:]), T-InputShapes[0,0]+1, axis=0), M-InputShapes[0,1]+1, axis=1)
        rate = np.r_[np.zeros((1, M)), rate]
    if pmt is not None:
        pmt = np.repeat(np.repeat(np.reshape(pmt, InputShapes[1,:]), T-InputShapes[1,0]+1, axis=0), M-InputShapes[1,1]+1, axis=1)
        if when=="end":
            pmt = np.r_[np.zeros((1, M)), pmt]
        elif when=="begin":
            pmt = np.r_[pmt, np.zeros((1, M))]
        else:
            raise Exception("'%s' is an unrecognized kind of when!" % (when, ))
    if pv is not None:
        pv = np.repeat(np.reshape(pv, InputShapes[2,1:]), M-InputShapes[2,1]+1, axis=0)
        if pmt is not None:
            pmt[0, :] += pv
    if fv is not None:
        fv = np.repeat(np.reshape(fv, InputShapes[3,1:]), M-InputShapes[3,1]+1, axis=0)
        if pmt is not None:
            pmt[-1, :] += fv
    return (ReturnScalar, rate, pmt, pv, fv)

# 计算现值
def pv(rate, pmt, fv=0, when="end", output="single"):
    """
    计算现值
    
    输入参数
    --------
     * rate: 每期收益率, array((T, M)), array((T, )), float
     * pmt: 每期现金流, array((T, M)), array((T, )), float
     * fv: 终值, array((M, )), float
     * when: 现金流 pmt 发生在每期的期初 (when = 'begin') 还是期末 (when = 'end')
     * output: 返回每期末的现值 (output = 'multi') 还是只返回 0 时刻的现值 (output = 'single')
    
    返回值
    ------
     * pv: 现值, 如果 output = "single": array((M, )), float; output = "multi": array((T+1, M)), array((T+1, ))
    
    示例
    ----
    一共 5 期, 每期投入资金 8 元, 每期的现金流发生在期末, 最后一期获得收入 100 元, 每期收益率从对数正态分布抽样, 求期初所需的资金投入
    >>> np.random.seed(0)
    >>> Rate = np.random.lognormal(-3.15, 0.55, (5, ))
    >>> pv(rate=Rate, pmt=-8, fv=100, when='end', output='single')
    -31.01020941024582
    
    一共 5 期, 每期投入资金 8 元, 每期的现金流发生在期末, 最后一期获得收入 100 元, 每期收益率从对数正态分布抽样, 模拟 3 次, 求期初所需的资金投入
    >>> np.random.seed(0)
    >>> Rate = np.random.lognormal(-3.15, 0.55, (5, 3))
    >>> pv(rate=Rate, pmt=-8, fv=100, when='end', output='single')
    array([-35.03724883, -41.43450797, -41.46922181])
    
    一共 5 期, 每期投入资金 8 元, 每期的现金流发生在期末, 最后一期获得收入 100 元, 每期收益率从对数正态分布抽样, 模拟 3 次, 如果从每期期初进行投资, 求每期期初所需的资金投入
    >>> np.random.seed(0)
    >>> Rate = np.random.lognormal(-3.15, 0.55, (5, 3))
    >>> pv(rate=Rate, pmt=-8, fv=100, when='end', output='multi')
    array([[-35.03724883, -41.43450797, -41.46922181],
           [-38.99881933, -43.64717928, -44.51348044],
           [-53.90636052, -57.8288741 , -53.82812598],
           [-66.37986257, -68.42445954, -64.33137127],
           [-78.37474607, -79.96941913, -79.22852185],
           [-92.        , -92.        , -92.        ]])
    """
    ReturnScalar, rate, pmt, _, fv = _adjust_input(rate=rate, pmt=pmt, pv=None, fv=fv, when=when)
    if output=="single":
        PV = - np.sum(pmt / np.cumprod(1 + rate, axis=0), axis=0)
    elif output=="multi":
        PV = np.full_like(rate, fill_value=np.nan)
        for i in range(rate.shape[0]):
            rate[:i+1, :] = 0
            PV[i, :] = - np.sum(pmt[i:, :] / np.cumprod(1 + rate[i:, :], axis=0), axis=0)
    else:
        raise Exception("'%s' is an unrecognized kind of output!" % (output, ))
    if ReturnScalar:
        PV = np.squeeze(PV)
        if np.ndim(PV)==0:
            return float(PV)
    return PV

# 计算终值
def fv(rate, pmt, pv=0, when="end", output="single"):
    """
    计算终值
    
    输入参数
    --------
     * rate: 每期收益率, array((T, M)), array((T, )), float
     * pmt: 每期现金流, array((T, M)), array((T, )), float
     * pv: 现值, array((M, )), float
     * when: 现金流 pmt 发生在每期的期初 (when = 'begin') 还是期末 (when = 'end')
     * output: 返回每期末的现值 (output = 'multi') 还是只返回 0 时刻的现值 (output = 'single')
    
    返回值
    ------
     * fv: 终值, 如果 output = "single": array((M, )), float; output = "multi": array((T+1, M)), array((T+1, ))
    
    示例
    ----
    一共 5 期, 每期投入资金 8 元, 每期的现金流发生在期末, 第一期初投入 10 元, 每期收益率从对数正态分布抽样, 求期末资金结余
    >>> np.random.seed(0)
    >>> Rate = np.random.lognormal(-3.15, 0.55, (5, ))
    >>> fv(rate=Rate, pmt=-8, pv=-10, when='end', output='single')
    66.19403293510359
    
    一共 5 期, 每期投入资金 8 元, 每期的现金流发生在期末, 第一期初投入 10 元, 每期收益率从对数正态分布抽样, 模拟 3 次, 求期末资金结余
    >>> np.random.seed(0)
    >>> Rate = np.random.lognormal(-3.15, 0.55, (5, 3))
    >>> fv(rate=Rate, pmt=-8, pv=-10, when='end', output='single')
    array([61.61309656, 57.8549273 , 58.41410733])
    
    一共 5 期, 每期投入资金 8 元, 每期的现金流发生在期末, 第一期初投入 10 元, 每期收益率从对数正态分布抽样, 模拟 3 次, 求每期期末资金结余
    >>> np.random.seed(0)
    >>> Rate = np.random.lognormal(-3.15, 0.55, (5, 3))
    >>> fv(rate=Rate, pmt=-8, pv=-10, when='end', output='multi')
    array([[10.        , 10.        , 10.        ],
           [19.14031037, 18.53526296, 18.73773797],
           [29.98349978, 28.77371738, 27.20500781],
           [40.16063307, 37.90806558, 36.30652693],
           [50.32276521, 47.66850477, 47.79315993],
           [61.61309656, 57.8549273 , 58.41410733]])
    """
    ReturnScalar, rate, pmt, pv, _ = _adjust_input(rate=rate, pmt=pmt, pv=pv, fv=None, when=when)
    if output=="single":
        FV = - np.sum(pmt * (np.prod(1 + rate, axis=0) / np.cumprod(1 + rate, axis=0)), axis=0)
    elif output=="multi":
        Coef = np.cumprod(1 + rate, axis=0)
        FV = np.full_like(rate, fill_value=np.nan)
        for i in range(rate.shape[0]):
            FV[i, :] = - np.sum(pmt[:i+1, :] * (Coef[i, :] / Coef[:i+1, :]), axis=0)
    else:
        raise Exception("'%s' is an unrecognized kind of output!" % (output, ))
    if ReturnScalar:
        FV = np.squeeze(FV)
        if np.ndim(FV)==0:
            return float(FV)
    return FV

# 计算每期现金流
def pmt(rate, pv, fv=0, when="end"):
    """
    计算每期现金流
    
    输入参数
    --------
     * rate: 每期收益率, array((T, M)), array((T, )), float
     * pv: 现值, array((M, )), float
     * fv: 终值, array((M, )), float
     * when: 现金流 pmt 发生在每期的期初 (when = 'begin') 还是期末 (when = 'end')
    
    返回值
    ------
     * pmt: 每期现金流, array((M, )), float
    
    示例
    ----
    一共 5 期, 第一期初投入 10 元, 期末获得收入 100 元, 每期收益率从对数正态分布抽样, 假设每期的现金流发生在期末, 求每期的资金投入
    >>> np.random.seed(0)
    >>> Rate = np.random.lognormal(-3.15, 0.55, (5, ))
    >>> pmt(rate=Rate, pv=-10, fv=100, when='end')
    -13.412040129971855
    
    一共 5 期, 第一期初投入 10 元, 期末获得收入 100 元, 每期收益率从对数正态分布抽样, 假设每期的现金流发生在期末, 模拟 3 次, 求每期的资金投入
    >>> np.random.seed(0)
    >>> Rate = np.random.lognormal(-3.15, 0.55, (5, 3))
    >>> pmt(rate=Rate, pv=-10, fv=100, when='end')
    array([-14.64645695, -15.58980491, -15.36440432])
    """
    ReturnScalar, rate, _, pv, fv = _adjust_input(rate=rate, pmt=None, pv=pv, fv=fv, when=when)
    rate = np.flipud(rate[1:])
    if when=="end":
        Pmt = - (pv * np.prod(1 + rate, axis=0) + fv) / np.sum(np.cumprod(1 + np.r_[np.zeros((1, rate.shape[1])), rate[:-1]], axis=0), axis=0)
    elif when=="begin":
        Pmt = - (pv * np.prod(1 + rate, axis=0) + fv) / np.sum(np.cumprod(1 + rate[1:], axis=0), axis=0)
    else:
        raise Exception("'%s' is an unrecognized kind of when!" % (when, ))
    if ReturnScalar:
        Pmt = np.squeeze(Pmt)
        if np.ndim(Pmt)==0:
            return float(Pmt)
    return Pmt

# 计算每期收益率
def rate(pmt, pv, fv=0, when="end"):
    """
    计算每期收益率
    
    输入参数
    --------
     * pmt: 每期现金流, array((T, M)), array((T, )), float
     * pv: 现值, array((M, )), float
     * fv: 终值, array((M, )), float
     * when: 现金流 pmt 发生在每期的期初 (when = 'begin') 还是期末 (when = 'end')
    
    返回值
    ------
     * rate: 每期收益率, array((M, )), float
    
    示例
    ----
    一共 5 期, 每期投入资金 8 元, 每期的现金流发生在期末, 第一期初投入 10 元, 期末的预期收入分别为 80,100,120 元, 求每期所需的平均收益率
    >>> rate(pmt=-8+np.zeros((5,)), pv=-10, fv=np.array([80,100,120]), when='end')
    array([0.17946938, 0.26501948, 0.3356946 ])
    """
    ReturnScalar, _, pmt, pv, fv = _adjust_input(rate=None, pmt=pmt, pv=pv, fv=fv, when=when)
    Rate = np.apply_along_axis(np.irr, axis=0, arr=pmt)
    if ReturnScalar:
        Rate = np.squeeze(Rate)
        if np.ndim(Rate)==0:
            return float(Rate)
    return Rate