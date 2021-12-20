import argparse 
import pandas as pd 

df = pd.read_csv("/research2/mtc/cp_traces/exclusive_eval/w106_2560.csv", 
    names=["c", "t1", "t2", "t1_r", "t1_w", "t2_r", "t2_w", "m_r", "m_w", "mean_lat", "hit_rate", "cost", "p_d"])

# remove values where there is no T1 or T2 cache 
indexNames = df[(df["t1"]==0) & (df["t2"]==0)].index
df.drop(indexNames , inplace=True)

df_combo1 = df[df["c"]=="D1_O1_H1"]
df_combo2 = df[df["c"]=="D1_S2_H1"]


main_df = df_combo1.merge(df_combo2, how="inner", on=["t1", "t2"])

diff_df = 100*(main_df["mean_lat_x"]-main_df["mean_lat_y"])/main_df["mean_lat_x"]

print(diff_df)
print(diff_df.describe())
print(main_df["c_x"].unique())


