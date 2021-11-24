import argparse 
import pathlib, json 
import pandas as pd 
import numpy as np 
import multiprocessing as mp

from mascots.mtCache.mtCache import MTCache
from mascots.traceAnalysis.rdHist import read_reuse_hist_file, bin_rdhist


WORKLOAD_DIR = pathlib.Path("/research2/mtc/cp_traces/rdhist/4k/")


def process_t1(df, t1_size_array):
    pass 


def get_t1_size_array(num_rows, cpu_count):
    remainder = (num_rows + 1) % cpu_count 
    t1_size_array_length = (num_rows + 1) + cpu_count - remainder 
    t1_size_array = np.empty(t1_size_array_length, dtype=int)
    t1_size_array.fill(-1)
    t1_size_array[:num_rows+1] = np.arange(0, num_rows+1)
    return t1_size_array.reshape(-1, cpu_count).transpose()


def main(workload_name):
    t1_size = 1
    bin_width = 2560
    workload_path = WORKLOAD_DIR.joinpath("{}.csv".format(workload_name))
    rd_hist = bin_rdhist(read_reuse_hist_file(workload_path), bin_width)

    print("Data Loaded! ... ")
    
    device_config_list = []
    for config_path in pathlib.Path("../../mascots/mtCache/device_config").iterdir():
        with config_path.open("r") as f:
            device_config_list.append(json.load(f))

    mt_cache = MTCache()
    data = mt_cache.analyze_t2_exclusive(rd_hist, t1_size, bin_width, device_config_list)
    print(data)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Generate exclusive cache analysis for a workload")
    # parser.add_argument("workload_name", help="The name of the workload to be evaluted")
    # args = parser.parse_args()

    main("w106")