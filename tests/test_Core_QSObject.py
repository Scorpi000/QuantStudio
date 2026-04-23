import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from multiprocess import Process, Pool

from QuantStudio.Core.QSObject import QSQueue, QSFileLock


# ---------------------- QSFileLock ---------------------
def testQSFileLockFunc(lock, i):
    # print(f"子进程 {i}: ", os.getpid())
    with lock:
        print(i, "lock acquired!")
        for j in range(5, -1, -1):
            time.sleep(1)
            print(i, f"倒计时 {j}")

if __name__=="__main__":
    print("主进程: ", os.getpid())

    lock = QSFileLock(proc_lock=None)
    nTask = 4

    # Procs = []
    # for i in range(nTask):
    #     Procs.append(Process(target=testQSFileLockFunc, args=(lock, i)))
    #     Procs[-1].start()
    # for iProc in Procs: iProc.join()

    # with Pool(processes=nTask) as Executor:
    #     Futures = []
    #     for i in range(nTask):
    #         Futures.append(Executor.apply_async(testQSFileLockFunc, (lock, i)))
    #     for iFuture in Futures:
    #         iFuture.get()

    # 不能用
    # with ProcessPoolExecutor(max_workers=nTask) as Executor:
    #     Futures = []
    #     for i in range(nTask):
    #         Futures.append(Executor.submit(testQSFileLockFunc, lock, i))
    #     for iFuture in Futures:
    #         iFuture.result()

    # with ThreadPoolExecutor(max_workers=nTask) as Executor:
    #     Futures = []
    #     for i in range(nTask):
    #         Futures.append(Executor.submit(testQSFileLockFunc, lock, i))
    #     for iFuture in Futures:
    #         iFuture.result()

    print("===")


# ---------------------- QSQueue ------------------------
def testQSQueueFunc(i, q):
    time.sleep(0.01)
    q.put((i, "a" * 10000000))

if __name__=="__main__1":
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