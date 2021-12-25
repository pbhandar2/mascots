import pandas as pd 

df = pd.read_csv("block_trace_stat.csv", 
    names=["w", "min_lba", "r", "tot_io_gb", "write_per", "x", "y", "z", "s", "read_size", "write_size", "time"])

df["read_size_kb"] = df["read_size"]/1024
df["write_size_kb"] = df["write_size"]/1024

print(df[["w", "read_size_kb", "write_size_kb"]])
for index,row in df.iterrows():
    print("{},{},{}".format(row["w"], row["read_size_kb"], row["write_size_kb"]))
