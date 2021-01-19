import numpy as np
import pandas as pd
set = []
for i in range(50):
    #if 16>i>10:
    #    continue
    a = np.load("rew1new78_{i}.npy".format(i=i)).flatten()
    set.append(a)
set = np.array(set)
a = pd.DataFrame(set.mean(axis=0), index=None, columns=['reward'])
a.to_csv('progress.csv')