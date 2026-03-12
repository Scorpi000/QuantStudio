import time
import numpy as np
import pandas as pd


# H5 文件中文路径问题
# with open('中文简称.h5', mode="a") as f: pass
# with pd.HDFStore('中文简称.h5') as jFile:
#     jFile["StdData"] = pd.DataFrame(np.random.randn(5, 2))
# with pd.HDFStore('中文简称.h5', mode="r") as jFile:
#     print(jFile["StdData"])

FileName = "中文简称"

# 写
df = pd.DataFrame(np.random.randn(10000, 10000))
startT = time.perf_counter()
df.to_feather(f"{FileName}.feather")
print("feather: ", time.perf_counter() - startT)

startT = time.perf_counter()
df.to_parquet(f"{FileName}.parquet")
print("parquet: ", time.perf_counter() - startT)

startT = time.perf_counter()
df.to_hdf(f"{FileName}.h5", key="aha")
print("hdf5: ", time.perf_counter() - startT)

# 读
startT = time.perf_counter()
df = pd.read_feather(f"{FileName}.feather")
print("feather: ", time.perf_counter() - startT)

startT = time.perf_counter()
df = pd.read_parquet(f"{FileName}.parquet")
print("parquet: ", time.perf_counter() - startT)

startT = time.perf_counter()
df = pd.read_hdf(f"{FileName}.h5", key="aha")
print("hdf5: ", time.perf_counter() - startT)