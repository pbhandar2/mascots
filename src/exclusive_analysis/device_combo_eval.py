import argparse 
import pathlib 
import math 
import numpy as np 
import pandas as pd 
from collections import defaultdict 
import matplotlib.pyplot as plt


def main(data_path):
    print(
        "Device combo evaluation for file: {}".format(data_path.name)
    )

    df = pd.read_csv(data_path, 
        names=["c", "t1", "t2", "t1_r", "t1_w", "t2_r", "t2_w", "m_r", "m_w", "mean_lat", "hit_rate", "cost", "p_d"])

    # remove values where there is no T1 or T2 cache 
    indexNames = df[(df["t1"]==0) & (df["t2"]==0)].index
    df.drop(indexNames, inplace=True)
    df.dropna(inplace=True)

    # for each device combo, generate a cost-latency curve, plot multiple of them in a single graph or a select of them 
    min_lat_data = {}
    max_lat, min_lat = None 
    plot_cache_size = math.inf 
    for key, data in df.groupby(["c"]):
        # get the storage and the cache label for this configuration 
        split_key = key.split("_")
        storage_label = split_key[-1]
        cache_label = "{}_{}".format(split_key[0], split_key[1])

        # only continue for HDD configs 
        if "H" not in storage_label:
            continue 

        # evaluate from $1 to max dollar for the workload and get the mean latency for different configuration
        min_lat_array = [] 
        max_cost = math.ceil(data["cost"].max())
        plot_cache_size = max_cost if max_cost < plot_cache_size else plot_cache_size 

        for i in range(1, math.ceil(data["cost"].max()+1)):
            cost_data = data[data["cost"]<i]
            min_lat_entry = cost_data[cost_data["mean_lat"]==cost_data["mean_lat"].min()]
            min_lat_array.append(min_lat_entry.iloc[0]["mean_lat"])


        # store the data according to the storage and cache label 
        if storage_label not in min_lat_data:
            min_lat_data[storage_label] = {
                cache_label: min_lat_array 
            }
        else:
            min_lat_data[storage_label][cache_label] = min_lat_array

    # for each device combination 
    cache_marker = {
        "D1_O1": "^",
        "D1_S1": "*",
        "D1_S2": "o"
    }

    cache_line = {
        "D1_O1": "-",
        "D1_S1": "-.",
        "D1_S2": ":"
    }

    for storage_label in min_lat_data:
        fig = plt.figure(figsize=[14, 10])
        plt.rcParams.update({'font.size': 25})
        ax = plt.subplot(1,1,1)

        for cache_label in min_lat_data[storage_label]:
            line_marker = "{}{}".format(cache_line[cache_label], cache_marker[cache_label])
            ax.plot(min_lat_data[storage_label][cache_label], line_marker, label=cache_label)
        
        ax.set_yscale("log")
        plt.legend() 
        plt.tight_layout()
        plt.savefig("lat_{}_{}.png".format(data_path.stem, storage_label))
        plt.close() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Determine the performance of each device for a workload")
    parser.add_argument("exclusive_cache_data_path", 
        type=pathlib.Path,
        help="Path to the file containing exclusive cache information")
    args = parser.parse_args()

    main(args.exclusive_cache_data_path)