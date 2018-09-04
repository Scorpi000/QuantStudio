# coding=utf-8
import os
import datetime as dt

import numpy as np
import pandas as pd
import h5py

#Tables = ["StyleValueFactor"]

MainDir = "C:\\HST\\HDF5Data"
for iTable in os.listdir(MainDir):
    iTablePath = MainDir+os.sep+iTable
    for jFactor in os.listdir(iTablePath):
        if jFactor.split(".")[-1]!="hdf5": continue
        with h5py.File(iTablePath+os.sep+jFactor) as File:
            if "DateTime" in File: continue
            ijDates = File["Date"][...]
            ijDTs = np.array([dt.datetime(int(kDate[:4]), int(kDate[4:6]), int(kDate[6:8]), 23, 59, 59, 999999).timestamp() for kDate in ijDates])
            File.create_dataset("DateTime", shape=(ijDTs.shape[0],), maxshape=(None,), data=ijDTs)
            del File["Date"]
    print(iTable)

#import QuantStudio.api as QS
#HDB = QS.FactorDB.HDF5DB()
#HDB.connect()
#FT = HDB.getTable("StyleValueFactor")
#DTs = FT.getDateTime(ifactor_name="BP_LR")
#IDs = FT.getID(ifactor_name="BP_LR")
#Data = FT.readData(factor_names=["BP_LR"])
#HDB.disconnect()