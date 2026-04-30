import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import h5py
import pandas as pd
import numpy as np
from multiprocess import Process, Pool


# ---------------------- FileLock ----------------------
from filelock import FileLock


def testFileLockFunc(file_path, i):
    lock = FileLock(file_path)
    with lock:
        print(i, "lock acquired!")
        for j in range(1, -1, -1):
            time.sleep(1)
            print(i, f"倒计时 {j}")

if __name__=="__main__":
    print("主进程: ", os.getpid())

    nTask = 6
    FilePath = r".\LockFile"

    # Procs = []
    # for i in range(nTask):
    #     Procs.append(Process(target=testFileLockFunc, args=(FilePath, i)))
    #     Procs[-1].start()
    # for iProc in Procs: iProc.join()

    # with Pool(processes=nTask) as Executor:
    #     Futures = []
    #     for i in range(nTask):
    #         Futures.append(Executor.apply_async(testFileLockFunc, (FilePath, i)))
    #     for iFuture in Futures:
    #         iFuture.get()

    # with ProcessPoolExecutor(max_workers=nTask) as Executor:
    #     Futures = []
    #     for i in range(nTask):
    #         Futures.append(Executor.submit(testFileLockFunc, FilePath, i))
    #     for iFuture in Futures:
    #         iFuture.result()

    with ThreadPoolExecutor(max_workers=nTask) as Executor:
        Futures = []
        for i in range(nTask):
            Futures.append(Executor.submit(testFileLockFunc, FilePath, i))
        for iFuture in Futures:
            iFuture.result()

    print("===")


# ---------------------- QSQueue ------------------------
def testQSQueueFunc(i, q):
    time.sleep(0.01)
    q.put((i, "a" * 10000000))

if __name__=="__main__1":
    from QuantStudio.Core.QSObject import QSQueue
    q = QSQueue(cache_size=1, batch_size=40)

    nProc = 4
    Procs = []
    for i in range(nProc):
        Procs.append(Process(target=testQSQueueFunc, args=(i, q)))
        Procs[-1].start()

    nFinished = 0
    while nFinished < nProc:
        i, iData = q.get()
        print(i, "finished, data length: ", len(iData))
        nFinished += 1
    for iProc in Procs: iProc.join()
    print("===")