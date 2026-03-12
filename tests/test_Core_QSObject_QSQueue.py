import time
from multiprocessing import Process
from QuantStudio.Core.QSObject import QSQueue


def test_func(i, q):
    time.sleep(0.01)
    q.put((i, "a" * 10000000))

if __name__=="__main__":
    q = QSQueue(cache_size=1, batch_size=40)

    nProc = 4
    Procs = []
    for i in range(nProc):
        Procs.append(Process(target=test_func, args=(i, q)))
        Procs[-1].start()

    nFinished = 0
    while nFinished < nProc:
        i, iData = q.get()
        print(i, "finished, data length: ", len(iData))
        nFinished += 1
    for iProc in Procs: iProc.join()
    print("===")