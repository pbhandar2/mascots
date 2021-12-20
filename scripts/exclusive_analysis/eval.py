import argparse 
import pathlib, json, time  
import traceback
import logging
import pandas as pd 
import numpy as np 
import multiprocessing as mp

from mascots.mtCache.mtCache import MTCache
from mascots.traceAnalysis.rdHist import read_reuse_hist_file, bin_rdhist

WORKLOAD_DIR = pathlib.Path("/research2/mtc/cp_traces/rdhist/4k/")
OUTPUT_DIR = pathlib.Path("/research2/mtc/cp_traces/mascots/exclusive")
DEVICE_LIST = []
for config_path in pathlib.Path("../../mascots/mtCache/device_config").iterdir():
    with config_path.open("r") as f:
        DEVICE_LIST.append(json.load(f))
START_TIME = time.time()

PAGE_SIZE = 4096 # 4KB
ALLOCATION_GRANULARITY = 10*1024*1024 # 10MB
MAX_TIER1 = 128*1024*1024*1024 # 256GB
MAX_TIER1_SIZE = int(MAX_TIER1/ALLOCATION_GRANULARITY)


def process_t1(rd_hist, t1_size_array, bin_width, device_config_list, output_dir):
    mt_cache = MTCache()
    for index, t1_size in enumerate(t1_size_array):
        if t1_size < 0:
            break 
        try:
            data = mt_cache.analyze_t2_exclusive(rd_hist, t1_size, bin_width, device_config_list)
            for cache_name in data:
                cache_output_dir = output_dir.joinpath(cache_name)
                cache_output_dir.mkdir(parents=True, exist_ok=True)
                output_file_path = cache_output_dir.joinpath("{}.csv".format(t1_size))
                data[cache_name].to_csv(output_file_path, index=False)
            cur_time = (time.time() - START_TIME)/60
        except:
            print("Error in {}, index={}, t1={}".format(rd_hist_path, index, t1_size))
            print(t1_size_array[index:])
            print(e)

        print("Done! Elasped: {}, Done: {}/{} .. {}".format(cur_time, index+1, len(t1_size_array), (index+1)/len(t1_size_array)))
        

def get_t1_size_array(num_rows, cpu_count):
    remainder = (num_rows + 1) % cpu_count 
    t1_size_array_length = (num_rows + 1) + cpu_count - remainder 
    t1_size_array = np.empty(t1_size_array_length, dtype=int)
    t1_size_array.fill(-1)
    t1_size_array[:num_rows+1] = np.arange(0, num_rows+1)
    return t1_size_array.reshape(-1, cpu_count).transpose()


def main(workload_name, bin_width, cpu_count):
    workload_output_dir = OUTPUT_DIR.joinpath(workload_name)
    workload_output_dir.mkdir(parents=True, exist_ok=True)

    rd_hist_path = WORKLOAD_DIR.joinpath("{}.csv".format(workload_name))
    rd_hist = bin_rdhist(read_reuse_hist_file(rd_hist_path), bin_width)
    num_rows = min(len(rd_hist), MAX_TIER1_SIZE) 
    print("RD Hist Loaded! {}".format(rd_hist_path))

    pool = mp.Pool(cpu_count)
    per_process_input_array = get_t1_size_array(num_rows, cpu_count)
    try:
        results=[pool.apply_async(process_t1,
            args=(rd_hist, 
                process_input, 
                bin_width,
                DEVICE_LIST,
                workload_output_dir)) for process_input in per_process_input_array]
    except Exception as e:
        print(traceback.format_exc())

    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate exclusive cache analysis for a workload")
    parser.add_argument("workload_name", help="The name of the workload to be evaluted")
    parser.add_argument("--c", type=int, default=4, help="The number of CPU to use for multiprocessing")
    args = parser.parse_args()

    main(args.workload_name, 2560, args.c)
    end = time.time()
    print("Time Elasped: {}".format((end - START_TIME)/60))