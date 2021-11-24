import argparse 
import pathlib 
import json 
import time 
import numpy as np 
import pandas as pd 

from mascots.traceAnalysis.rdHist import read_reuse_hist_file, bin_rdhist
from mascots.mtCache.greedyLatAllocater import GreedyLatAllocater

ALLOCATION_LOG_DIR = pathlib.Path("/research2/mtc/cp_traces/mascots/allocation_logs")
DATA_DIR = pathlib.Path("/research2/mtc/cp_traces/mascots/cost") 

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


def load_data(mt_label):
    """ Load the DataFrame of each workload and add the necessary columns """
    data = {}
    for workload_index, workload_name in enumerate(WORKLOAD_NAME_LIST):
        data_path = DATA_DIR.joinpath(workload_name, "{}.csv".format(mt_label))
        df = pd.read_csv(data_path)
        df.loc[:, "lat_reduced_wb"] = df["wb_min_lat"].shift(1, fill_value=0)-df["wb_min_lat"]
        df.loc[df["lat_reduced_wb"]<0, "lat_reduced_wb"] = 0
        df.loc[:, "cum_lat_reduced_wb"] = df["lat_reduced_wb"].cumsum()
        df.loc[:, "cum_lat_reduced_wb_per_dollar"] = df.loc[:, "cum_lat_reduced_wb"]/df.loc[:, "c"]
        df["cum_lat_reduced_wb_per_dollar"].fillna(0, inplace=True)

        df.loc[:, "lat_reduced_wt"] = df["wt_min_lat"].shift(1, fill_value=0)-df["wt_min_lat"]
        df.loc[df["lat_reduced_wt"]<0, "lat_reduced_wt"] = 0
        df.loc[:, "cum_lat_reduced_wt"] = df["lat_reduced_wt"].cumsum()
        df.loc[:, "cum_lat_reduced_wt_per_dollar"] = df.loc[:, "cum_lat_reduced_wt"]/df.loc[:, "c"]
        df["cum_lat_reduced_wt_per_dollar"].fillna(0, inplace=True)
        data[workload_name] = {
            "df": df
        }
    return data 


def main():
    WRITE_TYPE="wb"
    for device_config in DEVICE_LIST:
        mt_label = "_".join([_["label"] for _ in device_config])
        data = load_data(mt_label)
        t1_size = int((device_config[0]["size"]*1024)/10) 
        t2_size = int((device_config[1]["size"]*1024)/10) 
        mt_size_array = [t1_size, t2_size]

        allocater = GreedyLatAllocater(WORKLOAD_NAME_LIST, device_config, mt_size_array, 
            data, pathlib.Path("./greedy_lat_data/{}_{}_greedy_lat_alloc.csv".format(WRITE_TYPE, mt_label)), mt_type=WRITE_TYPE)

        allocater.run()

        summary_path = pathlib.Path("./greedy_lat_data/{}_{}_greedy_lat_alloc_summary.csv".format(WRITE_TYPE, mt_label))
        allocater.get_latency(summary_path)




    # for device_config in DEVICE_LIST:
    #     mt_label = "_".join([_["label"] for _ in device_config])
    #     for metric_name in ["hits_per_size"]:
    #         t1_size = int((device_config[0]["size"]*1024)/10) 
    #         t2_size = int((device_config[1]["size"]*1024)/10) 
    #         mt_size_array = [t1_size, t2_size]

    #         # allocation_file_path = ALLOCATION_LOG_DIR.joinpath("{}.csv".format(mt_label))
    #         allocater = GreedyHitRateAllocater(WORKLOAD_NAME_LIST, device_config, mt_size_array, 
    #             data, pathlib.Path("{}_greedy_lat_alloc.csv".format(mt_label)))
    #         allocater.run(metric_name)

    #         summary_path = pathlib.Path("{}_greedy_lat_alloc_summary.csv".format(mt_label))
    #         allocater.get_latency(summary_path)


if __name__ == "__main__":
    main()

