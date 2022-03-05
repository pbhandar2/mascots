import argparse, json, pathlib, math 
from asyncore import dispatcher_with_send 
import pandas as pd 
import numpy as np 
from tabulate import tabulate
from collections import OrderedDict, defaultdict

from get_opt_vs_algo import OPT_VS_ALGO_HEADER_LIST
from mascots.traceAnalysis.RDHistProfiler import OPT_ROW_JSON 

with open("../../experiment_config.json") as f:
    config = json.load(f)

cache_name_replace_map = {
    "D1": "FastDRAM",
    "D3": "SlowDRAM",
    "H3": "SlowHDD",
    "S5": "SlowSSD",
    "S3": "MediumSSD",
    "S1": "FastSSD",
    "H1": "FastHDD"
}

num_points_array = [5, 10, 50]
num_points_array_2 = [1, 5, 10, 20]

def get_per_workload_cost(write_policy, step_size, num_points_array):
    final_out = {}
    max_points = np.array(num_points_array).max()
    for mt_label in config["priority_2_tier_mt_label_list"]:
        per_workload_cost_dict = {}
        error_dir = pathlib.Path(config["error_per_mt_config"]).joinpath(mt_label, write_policy, str(step_size), str(max_points))
        for workload_path in error_dir.iterdir():
            workload_name = workload_path.stem
            df = pd.read_csv(workload_path, names=OPT_VS_ALGO_HEADER_LIST)
            filter_df = df[df["opt_type"]!=0]
            if len(filter_df) > 0:
                filter_df = filter_df.drop_duplicates()
                per_workload_cost_dict[workload_name] = filter_df["cost"].to_numpy().min()
        final_out[mt_label] = per_workload_cost_dict
    return final_out 

def get_labels(mt_label):
    cache_name_replace_map = {
        "D1": "FastDRAM",
        "D3": "SlowDRAM",
        "H3": "SlowHDD",
        "S5": "SlowSSD",
        "S3": "MediumSSD",
        "S1": "FastSSD",
        "H1": "FastHDD"
    }

    split_mt_label = mt_label.split("_")
    new_label = [cache_name_replace_map[_] for _ in split_mt_label]
    return "-".join(new_label)


def main3(write_policy, num_points_array=[5, 10, 50]):
    max_cost_df = pd.read_csv("../../workload_analysis/max_cost.csv", 
                                delimiter=" ",
                                names=["workload", "max_cost"])
    per_workload_min_cost = get_per_workload_cost(write_policy, 10, num_points_array)
    percent_eval_array = [5,10,20,50]
    for step_size in [1, 5, 10, 50]:
        lat_error_list = []
        # For each MT label 
        for mt_label in config["priority_2_tier_mt_label_list"]:
            lat_error_dict = defaultdict(list)
            mt_miss_dict = defaultdict(int)
            total_dict = defaultdict(int)

            error_dir = pathlib.Path(config["error_per_mt_config"]).joinpath(mt_label, write_policy, str(step_size), str(10))

            # for each workload 
            for workload_path in error_dir.iterdir():
                workload_name = workload_path.stem

                max_cost = int(max_cost_df[max_cost_df["workload"]==workload_name].iloc[0]["max_cost"])
                for percent_eval in percent_eval_array:
                    num_points = int((percent_eval/100)*max_cost)
                    num_points = num_points - num_points % 10 
                    if num_points == 0:
                        num_points = 5
                    error_path = pathlib.Path(config["error_per_mt_config"]).joinpath(mt_label, write_policy, str(step_size), str(num_points),
                                                                                        "{}.csv".format(workload_name))

                    if not error_path.exists():
                        continue 

                    df = pd.read_csv(error_path, names=OPT_VS_ALGO_HEADER_LIST)
                    filter_df = df[df["opt_type"]!=0]
                    filter_df = filter_df.drop_duplicates()
                    
                    for _, row in filter_df.iterrows():
                        cur_cost = int(row["cost"])
                        if workload_name in per_workload_min_cost[mt_label]:
                            if cur_cost >= per_workload_min_cost[mt_label][workload_name]:
                                if row["hmrc_type"] != 0:
                                    lat_error_dict[percent_eval].append(row["error"])
                                else:
                                    mt_miss_dict[percent_eval] += 1
                                total_dict[percent_eval] += 1

        print(mt_label, lat_error_dict.keys())
        lat_error_entry = [mt_label]
        for percent_eval in percent_eval_array:
            mean_error = np.array(lat_error_dict[percent_eval], dtype=float).mean()
            lat_error_entry.append(int(np.ceil(mean_error)))
        lat_error_list.append(lat_error_entry)

        print(tabulate(lat_error_list, headers=["MT"] + percent_eval_array))


def get_potential_gain(write_policy, mt_label, cost, workload_name):
    ex_path = pathlib.Path(config["ex_cost_analysis_dir"]).joinpath(mt_label, write_policy, 
                                                                        "{}.csv".format(workload_name))
    ex_df = pd.read_csv(ex_path, names=OPT_ROW_JSON)
    opt_row = ex_df[ex_df["cost"]==cost].iloc[0]
    st_latency = opt_row["st_latency"]
    mt_p_latency = opt_row["mt_p_latency"]
    mt_np_latency = opt_row["mt_np_latency"]
    max_eval = opt_row["st_size"]
    potential = 100*(st_latency-min(mt_p_latency, mt_np_latency))/st_latency
    assert(potential>0)
    return potential, max_eval


def main4(write_policy, step_size, num_points_array=[5, 10, 50]):
    per_workload_min_cost = get_per_workload_cost(write_policy, step_size, num_points_array)
    entry_array = []
    mt_miss_array = []
    for mt_label in config["priority_2_tier_mt_label_list"]:
        error_dict = defaultdict(list)
        space_evaluated_dict = defaultdict(list)
        mt_miss_dict = defaultdict(int)
        total_dict = defaultdict(int)
        mt_miss_lat_potential_dict = defaultdict(list)
        mt_hit_lat_potential_dict = defaultdict(list)
        for num_points in num_points_array:
            error_dir = pathlib.Path(config["error_per_mt_config"]).joinpath(mt_label, write_policy, str(step_size), str(num_points))
            for workload_path in error_dir.iterdir():
                workload_name = workload_path.stem
                df = pd.read_csv(workload_path, names=OPT_VS_ALGO_HEADER_LIST)
                filter_df = df[df["opt_type"]!=0]
                filter_df = filter_df.drop_duplicates()
                for _, row in filter_df.iterrows():
                    if np.isnan(row["cost"]):
                        continue 

                    cur_cost = int(row["cost"])
                    if workload_name in per_workload_min_cost[mt_label]:
                        if cur_cost >= per_workload_min_cost[mt_label][workload_name]:
                            potential_gain, st_size = get_potential_gain(write_policy, mt_label, cur_cost, workload_name)
                            if row["hmrc_type"] != 0:
                                error_dict[num_points].append(row["error"])
                                mt_hit_lat_potential_dict[num_points].append(potential_gain)
                                space_evaluated_dict[num_points].append(100*10/st_size)
                            else:
                                mt_miss_dict[num_points] += 1
                                mt_miss_lat_potential_dict[num_points].append(potential_gain)
                            total_dict[num_points] += 1
            print("Evaluated: {}".format(mt_label))

        error_table_header = ["MT"]
        entry = [mt_label]
        for num_points in num_points_array:
            mean_error = np.array(error_dict[num_points], dtype=float).mean()
            std_error = np.array(error_dict[num_points], dtype=float).std()
            mean_space_evaluted = np.array(space_evaluated_dict[num_points], dtype=float).mean()
            entry.append(mean_error)
            entry.append(std_error)
            entry.append(mean_space_evaluted)
            error_table_header.append("mean-{}".format(str(num_points)))
            error_table_header.append("std-{}".format(str(num_points)))
            error_table_header.append("space-{}".format(str(num_points)))
        entry_array.append(entry)

        mt_miss_entry = [mt_label]
        for num_points in num_points_array:
            mt_miss_entry.append(100*mt_miss_dict[num_points]/total_dict[num_points])
            mt_miss_entry.append(np.array(mt_hit_lat_potential_dict[num_points]).mean())
            mt_miss_entry.append(np.array(mt_miss_lat_potential_dict[num_points]).mean())
        mt_miss_array.append(mt_miss_entry)
        
    min_num_points = min(num_points_array)
    max_num_points = max(num_points_array)
    out_file_name = "err_{}_{}_{}_{}.csv".format(write_policy, step_size, min_num_points, max_num_points)
    out_path = pathlib.Path(config["error_per_step_size"]).joinpath(out_file_name)
    with open(out_path, "w+") as f:
        for e in entry_array:
            cur_label = get_labels(e[0])
            row_str_array = cur_label.split("-")
            for _ in e[1:]:
                row_str_array.append(str(_))
            row_str = ",".join(row_str_array)
            f.write("{}\n".format(row_str))

    out_file_name = "mt_miss_{}_{}_{}_{}.csv".format(write_policy, step_size, min_num_points, max_num_points)
    out_path = pathlib.Path(config["error_per_step_size"]).joinpath(out_file_name)
    with open(out_path, "w+") as f:
        for e in mt_miss_array:
            cur_label = get_labels(e[0])
            row_str_array = cur_label.split("-")
            for _ in e[1:]:
                row_str_array.append(str(_))
            row_str = ",".join(row_str_array)
            f.write("{}\n".format(row_str))

    mt_miss_header = ["MT", 5, "5h", "5m", 10, "10h", "10m", 50, "50h", "50m"]
    print("wp: {} step: {}".format(write_policy, step_size))
    print("Mean area evaluated {}".format(np.array([_[6] for _ in entry_array]).mean()))
    print("Mean error: {}".format(np.array([_[4] for _ in entry_array]).mean()))
    print("Mean classification error {}".format(np.array([_[4] for _ in mt_miss_array]).mean()))
    print("Mean hit {}".format(np.array([_[5] for _ in mt_miss_array]).mean()))
    print("Mean miss {}".format(np.array([_[6] for _ in mt_miss_array]).mean()))
    print(tabulate(entry_array, headers=error_table_header))
    print(tabulate(mt_miss_array, headers=mt_miss_header))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mean and max percent error")
    parser.add_argument("--alg", 
        default="v2",
        help="The algorithm to analyze")
    parser.add_argument("--step_size", 
        default=1,
        type=int,
        help="Step size for analysis")
    parser.add_argument("--test",
        default=0,
        type=int,
        help="test falg")
    parser.add_argument("--wp",
        default="wb",
        help="test falg")
    args = parser.parse_args()



    for step_size in [10, 50, 5]:
        for wp in ["wb", "wt"]:
            main4(wp, step_size)

    # main3("wb", num_points_array)
