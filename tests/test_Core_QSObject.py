import time
from multiprocessing import Process
from QuantStudio.Core.QSObject import QSQueue, QSFileLock



# ---------------------- QSFileLock ---------------------
def testQSFileLockFunc(lock):
    with lock:
        print("aha")
        time.sleep(3)

if __name__=="__main__1":
    lock = QSFileLock()

    nProc = 4
    Procs = []
    for i in range(nProc):
        Procs.append(Process(target=testQSFileLockFunc, args=(lock, )))
        Procs[-1].start()

    for iProc in Procs: iProc.join()
    print("===")


# ---------------------- QSQueue ------------------------
def testQSQueueFunc(i, q):
    time.sleep(0.01)
    q.put((i, "a" * 10000000))

if __name__=="__main__":
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