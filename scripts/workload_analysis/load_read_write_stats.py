import argparse 
import pathlib 
import pandas as pd 

if __name__ == "__main__":
    data_path = pathlib.Path("/research2/mtc/cp_traces/general/block_read_write_stats.csv")

    df = pd.read_csv(data_path)
    df["r_w"] = df["read_count"]/df["write_count"]
    
    write_heavy_sorted_df = df.sort_values(by=["r_w"]).iloc[:10]
    read_heavy_sorted_df = df.sort_values(by=["r_w"]).iloc[-10:]

    # print(df.iloc[(df['r_w']-2).abs().argsort()[:10]])
    # print(write_heavy_sorted_df)
    # print(read_heavy_sorted_df)


    print(df["r_w"].describe())

    df["total_ws"] = df["total_ws"]/(256*1024)
    print(df["total_ws"].describe())

    df["read_ws"] = df["read_ws"]/(256*1024)
    print(df["read_ws"].describe())

    df["write_ws"] = df["write_ws"]/(256*1024)
    print(df["write_ws"].describe())
