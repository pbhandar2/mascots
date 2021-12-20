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
OUTPUT_DIR = pathlib.Path("/research2/mtc/cp_traces/mascots/cost")
DEVICE_LIST = []
for config_path in pathlib.Path("../../mascots/mtCache/device_config").iterdir():
    with config_path.open("r") as f:
        DEVICE_LIST.append(json.load(f))
START_TIME = time.time()

def process_cost(rd_hist, cost_array, bin_width, device_config):
    mt_cache = MTCache()
    data_array = []
    for cost in cost_array:
        if cost > -1:
            data = mt_cache.get_opt_exclusive_at_cost(rd_hist, bin_width, device_config, cost)
            data_array.append(data)
    return data_array


def get_cost_array(num_rows, cpu_count):
    remainder = (num_rows + 1) % cpu_count 
    cost_array_length = (num_rows + 1) + cpu_count - remainder 
    cost_array = np.empty(cost_array_length, dtype=int)
    cost_array.fill(-1)
    cost_array[:num_rows+1] = np.arange(0, num_rows+1)
    return cost_array.reshape(-1, cpu_count).transpose()



def main(workload_name, bin_width, cpu_count):
    workload_output_dir = OUTPUT_DIR.joinpath(workload_name)
    workload_output_dir.mkdir(parents=True, exist_ok=True)

    rd_hist_path = WORKLOAD_DIR.joinpath("{}.csv".format(workload_name))
    rd_hist = bin_rdhist(read_reuse_hist_file(rd_hist_path), bin_width)


    for device_config in DEVICE_LIST:
        max_cache_size = len(rd_hist) - 1
        max_cost = int(device_config[0]["price"]*max_cache_size*bin_width)
        pool = mp.Pool(cpu_count)
        per_process_input_array = get_cost_array(min(max_cost, 150), cpu_count)
        results=[pool.apply_async(process_cost,
            args=(rd_hist, 
                process_input, 
                bin_width,
                device_config)) for process_input in per_process_input_array]

        output_array = []
        for apply_result_obj in results:
            for output in apply_result_obj.get():
                output_array.append(output)

        df = pd.DataFrame(output_array)
        df = df.sort_values(by=["c"])
        df.to_csv(workload_output_dir.joinpath("{}.csv".format("_".join([_["label"] for _ in device_config]))), index=False)

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