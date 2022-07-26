# coding=utf-8
"""交易相关函数"""
import numpy as np

# 以先进先出的方式匹配交易
# vols: array((N, )), 按照时间升序排列的交易量
# ls: array((N, )), 按照时间升序排列的多空标识, True: 多, False: 空
# 返回
# 匹配交易量: array((M, ))
# 匹配交易对应的多头交易位置索引: array((M, ))
# 匹配交易对应的空头交易位置索引: array((M, ))
# 剩余未匹配的交易量: array((K, ))
# 剩余未匹配的交易位置索引: array((K, ))
# 剩余未匹配交易的多空标识: True or False
def matchTransactionFIFO(vols, ls):
    lvols, svols = vols[ls], vols[~ls]
    lidxs = np.arange(vols.shape[0])
    lidxs, sidxs = lidxs[ls], lidxs[~ls]
    lidx, sidx = 0, 0
    mvols, mlidxs, msidxs = [], [], []
    while (lidx<lvols.shape[0]) and (sidx<svols.shape[0]):
        itvol = min(lvols[lidx], svols[sidx])
        lvols[lidx] -= itvol
        svols[sidx] -= itvol
        mvols.append(itvol)
        mlidxs.append(lidxs[lidx])
        msidxs.append(sidxs[sidx])
        lidx += (lvols[lidx]<=0)
        sidx += (svols[sidx]<=0)
    return np.array(mvols), np.array(mlidxs, dtype=int), np.array(msidxs, dtype=int), np.r_[lvols[lidx:], svols[sidx:]], np.r_[lidxs[lidx:], sidxs[sidx:]], (lidx<lvols.shape[0])

# 以先进后出的方式匹配交易
# vols: array((N, )), 按照时间升序排列的交易量
# ls: array((N, )), 按照时间升序排列的多空标识, True: 多, False: 空
# 返回
# 匹配交易量: array((M, ))
# 匹配交易对应的多头交易位置索引: array((M, ))
# 匹配交易对应的空头交易位置索引: array((M, ))
# 剩余未匹配的交易量: array((K, ))
# 剩余未匹配的交易位置索引: array((K, ))
# 剩余未匹配交易的多空标识: True or False
def matchTransactionFILO(vols, ls):
    lvols, svols = [], []
    lidxs, sidxs = [], []
    mvols, mlidxs, msidxs = [], [], []
    for i, iVol in enumerate(vols):
        if ls[i]:# 多头交易
            while svols and (iVol>0):
                itvol = min(svols[-1], iVol)
                mvols.append(itvol)
                mlidxs.append(i)
                msidxs.append(sidxs[-1])
                svols[-1] -= itvol
                iVol -= itvol
                if svols[-1]<=0:
                    svols.pop()
                    sidxs.pop()
            if iVol>0:
                lvols.append(iVol)
                lidxs.append(i)
        else:# 空头交易
            while lvols and (iVol>0):
                itvol = min(lvols[-1], iVol)
                mvols.append(itvol)
                mlidxs.append(lidxs[-1])
                msidxs.append(i)
                lvols[-1] -= itvol
                iVol -= itvol
                if lvols[-1]<=0:
                    lvols.pop()
                    lidxs.pop()
            if iVol>0:
                svols.append(iVol)
                sidxs.append(i)
    return np.array(mvols), np.array(mlidxs, dtype=int), np.array(msidxs, dtype=int), np.array(lvols+svols), np.array(lidxs+sidxs), (lidxs!=[])

if __name__=="__main__":
    Vols = np.array([1, 3, 3, 2, 2])
    LS = (np.array([1, 1, -1, 1, -1])==1)
    mVols, LIdx, SIdx, nVols, nLS = matchTransactionFIFO(Vols, LS)
    print("mVols: ", mVols)
    print("LIdx: ", LIdx)
    print("SIdx: ", SIdx)
    print("nVols: ", nVols)
    print("nLS: ", nLS)
    print("===")
    mVols, LIdx, SIdx, nVols, nLS = matchTransactionFILO(Vols, LS)
    print("mVols: ", mVols)
    print("LIdx: ", LIdx)
    print("SIdx: ", SIdx)
    print("nVols: ", nVols)
    print("nLS: ", nLS)
    
    print("===")