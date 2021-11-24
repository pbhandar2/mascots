import argparse 
import pathlib 
import json 
import time 
import numpy as np 
import pandas as pd 

from mascots.traceAnalysis.rdHist import read_reuse_hist_file, bin_rdhist
from mascots.mtCache.greedyHitRateAllocater import GreedyHitRateAllocater

ALLOCATION_LOG_DIR = pathlib.Path("/research2/mtc/cp_traces/mascots/allocation_logs")
DATA_DIR = pathlib.Path("/research2/mtc/cp_traces/rdhist/4k") 

BIN_WIDTH = 2560

# Load all the different device types 
DEVICE_LIST = []
for config_path in pathlib.Path("../../mascots/mtCache/device_config").iterdir():
    with config_path.open("r") as f:
        DEVICE_LIST.append(json.load(f))

# Load all the workload names 
WORKLOAD_NAME_LIST = []
for _ in range(1,107):
    workload_name = "w{}".format(_)
    if _ < 10:
        workload_name = "w0{}".format(_)
    WORKLOAD_NAME_LIST.append(workload_name)


def load_data():
    """ Load the DataFrame of each workload and add the necessary columns """
    data = {}
    for workload_index, workload_name in enumerate(WORKLOAD_NAME_LIST):
        start_time = time.time() 
        rdhist_path = DATA_DIR.joinpath("{}.csv".format(workload_name))
        rd_hist = bin_rdhist(read_reuse_hist_file(rdhist_path), BIN_WIDTH)
        df = pd.DataFrame(rd_hist[1:], columns=["r", "w"])
        df["rd"] = df.index
        df["total_hits"] = df["r"] + df["w"]
        df["cum_total_hits"] = df["total_hits"].cumsum()
        df["cum_read_hits"] = df["r"].cumsum()

        df["hits_per_size"] = df["cum_total_hits"]/(df.index+1)
        df["read_hits_per_size"] = df["cum_read_hits"]/(df.index+1)
        data[workload_name] = {
            "df": df,
            "hist": rd_hist
        }
        end_time = time.time() 
        print("{} loaded, {} seconds".format(rdhist_path.name, end_time-start_time))
    return data 


def main():
    data = load_data()
    for device_config in DEVICE_LIST:
        mt_label = "_".join([_["label"] for _ in device_config])
        for metric_name in ["read_hits_per_size"]:
            t1_size = int((device_config[0]["size"]*1024)/10) 
            t2_size = int((device_config[1]["size"]*1024)/10) 
            mt_size_array = [t1_size, t2_size]

            allocater = GreedyHitRateAllocater(WORKLOAD_NAME_LIST, device_config, mt_size_array, 
                data, pathlib.Path("./greedy_hr_data/{}_{}_greedy_alloc.csv".format(mt_label, metric_name)))
            allocater.run(metric_name)

            summary_path = pathlib.Path("./greedy_hr_data/{}_{}_greedy_alloc_summary.csv".format(mt_label, metric_name))
            allocater.get_latency(summary_path)


if __name__ == "__main__":
    main()

