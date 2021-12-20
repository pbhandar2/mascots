import argparse 
import pathlib
import pandas as pd  
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np 


class PerDeviceComboStat:
    def __init__(self, name):
        self.name = name 
        self.num_config = 0
        self.num_wt_cache_opt = 0 
        self.num_wb_cache_opt = 0 
        self.wb_cache_opt_cost_counter = Counter()
        self.wt_cache_opt_cost_counter = Counter() 
        self.wt_lat_diff = 0 
        self.wb_lat_diff = 0 


def get_workload_dirs(data_path):
    workload_dirs_array = []
    for workload_dir in data_path.iterdir():
        workload_num = int(workload_dir.name.split("w")[-1])
        if workload_num > 50:
            workload_dirs_array.append(workload_dir)
    return workload_dirs_array


def plot_hist(bin_array, data_array):
    print(bin_array)
    print(data_array)


def main(data_path):
    workload_dir_list = get_workload_dirs(data_path)
    per_device_stat_dict = {}

    # inside the workload dir, there is a file for each device combination 
    for workload_dir in workload_dir_list:
        for device_combo_file in workload_dir.iterdir():
            if "st" not in device_combo_file.stem:

                mt_df = pd.read_csv(device_combo_file, 
                    names=["c", "wb_t1", "wb_t2", "wb_lat", "wb_d_lat", 
                        "wb_d_lat_per_dollar", "wt_t1", "wt_t2", "wt_lat", "wt_d_lat", "wt_d_lat_per_dollar"])

                # reformat file name to get the correspondinf ST data file 
                device_config_name = device_combo_file.stem
                st_df_path = workload_dir.joinpath("st_{}.csv".format(device_config_name))
                st_df = pd.read_csv(st_df_path,
                    names=["c", "s", "lat", "d_lat", "d_lat_per_dollar"])

                # join the MT and ST dataframe on the cost column 
                merged_df = pd.merge(mt_df, st_df, on="c")

                # check if device config exists, else create 
                if device_config_name not in per_device_stat_dict:
                    per_device_stat_dict[device_config_name] = PerDeviceComboStat(device_config_name)

                # update the statistics by iterating through each row 
                for index, row in merged_df.iterrows():
                    per_device_stat_dict[device_config_name].num_config += 1

                    if row["lat"]>row["wt_lat"]:
                        per_device_stat_dict[device_config_name].num_wt_cache_opt += 1
                        per_device_stat_dict[device_config_name].wt_cache_opt_cost_counter[row["c"]] += 1
                        per_device_stat_dict[device_config_name].wt_lat_diff += row["lat"] - row["wt_lat"]

                    if row["lat"]>row["wb_lat"]:
                        per_device_stat_dict[device_config_name].num_wb_cache_opt += 1
                        per_device_stat_dict[device_config_name].wb_cache_opt_cost_counter[row["c"]] += 1
                        per_device_stat_dict[device_config_name].wb_lat_diff += row["lat"] - row["wb_lat"]

    # now take the array of PerDeviceComboStat and process each
    label_array = []
    percent_wb_opt_array = []
    percent_wt_opt_array = []
    mean_wb_latency_reduced_list = []
    mean_wt_latency_reduced_list = []
    for device_name in per_device_stat_dict:
        label_array.append("{}_{}".format(device_name.split("_")[1], device_name.split("_")[2]))
        cur_data = per_device_stat_dict[device_name]
        percent_wb_opt_array.append(cur_data.num_wb_cache_opt/cur_data.num_config)
        percent_wt_opt_array.append(cur_data.num_wt_cache_opt/cur_data.num_config)

        mean_wb_latency_reduced_list.append(cur_data.wb_lat_diff/cur_data.num_wb_cache_opt)
        mean_wt_latency_reduced_list.append(cur_data.wt_lat_diff/cur_data.num_wt_cache_opt)

    
    wb_wt_reduced = np.zeros(shape=(len(mean_wb_latency_reduced_list), 3))
    wb_wt_reduced[:,0] = mean_wb_latency_reduced_list
    wb_wt_reduced[:,1] = mean_wt_latency_reduced_list
    wb_wt_reduced[:,2] = range(len(mean_wb_latency_reduced_list))
    wb_wt_reduced=wb_wt_reduced[wb_wt_reduced[:, 0].argsort()]

    # plot histogram of percentage configuration ST vs MT 
    plt.figure(figsize=[14, 10])
    plt.rcParams.update({'font.size': 30})
    ax = plt.subplot(1,1,1)

    x = np.arange(len(label_array))
    width = 0.35  # the width of the bars
    
    ax.bar(x-width/2, wb_wt_reduced[:,0], width, label="Write Through", hatch="--")
    ax.bar(x+width/2, wb_wt_reduced[:,1], width, label="Write Back", hatch="/")
    ax.set_xlabel("Device Combination")
    ax.set_ylabel("log(Mean Latency Reduced (ms))")
    ax.set_xticks(x)
    ax.set_xticklabels([label_array[int(_[2])] for _ in wb_wt_reduced], rotation=90)
    ax.set_yscale("log")

    plt.legend()
    plt.tight_layout()
    plt.savefig("st_v_mt.png")
    plt.close()

                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ST vs MT stats for various devices")
    parser.add_argument("--d", type=pathlib.Path, 
        default=pathlib.Path("/research2/mtc/cp_traces/exclusive_cost_data/4k"),
        help="Directory containing the exclusive cache results")
    args = parser.parse_args()
    main(args.d)