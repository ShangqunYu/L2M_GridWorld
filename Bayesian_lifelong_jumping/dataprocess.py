import numpy as np
import pandas as pd
import os
set = []
# for i in range(1):
    #if 5>i:
    #    continue


data = pd.read_csv("F://bayesianlifelongrl//output//cheetah-bodyparts//2021_06_07_04_23_03//progress.csv")


for i in range(60):
    data_t = data[data["Epoch"]>=(i*100)]
    data_f = data_t[data_t["Epoch"] < ((i + 1) * 100)]
    data_f["Epoch"] = data_f["Epoch"] - i*100
    os.mkdir("./output/back_hcbody_{i}".format(i=i))
    data_f.to_csv('./output/back_hcbody_{i}/progress.csv'.format(i=i))
