import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

df = pd.read_csv("/research2/mtc/cp_traces/exclusive_eval/w106_2560.csv", 
    names=["c", "t1", "t2", "t1_r", "t1_w", "t2_r", "t2_w", "m_r", "m_w", "mean_lat", "hit_rate", "cost", "p_d"])

# remove values where there is no T1 or T2 cache 
indexNames = df[(df["t1"]==0) & (df["t2"]==0)].index
df.drop(indexNames, inplace=True)
df.dropna(inplace=True)









# fig, ax = plt.subplots()
# im = ax.imshow(df[["t1", "t2", "mean_lat"]].to_numpy())
# plt.savefig("heat.png")