import pathlib
import matplotlib.pyplot as plt 
import pandas as pd 

data_path = pathlib.Path("/research2/mtc/cp_traces/general/block_read_write_stats.csv")

df = pd.read_csv(data_path)
df["read_ratio"] = df["read_count"]/df["write_count"]

plt.figure(figsize=[14, 10])
plt.rcParams.update({'font.size': 30})
ax = plt.subplot(1,1,1)

ax.boxplot(df["read_ratio"], vert=False, positions=[0], widths=[0.75])

plt.tight_layout()
plt.savefig("box.png")

