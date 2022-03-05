import argparse 
import pathlib, math
import pandas as pd 
import numpy as np 
from collections import Counter, defaultdict
from scipy import stats
import matplotlib.pyplot as plt

from mascots.traceAnalysis.RDHistProfiler import OPT_ROW_JSON 

COST_DATA_DIR = pathlib.Path("/research2/mtc/cp_traces/exhaustive_cost_analysis/")
PER_ROW_DATA_JSON = {
    "cost": 0,
    "st_opt_count": 0,
    "mt_p_opt_count": 0,
    "mt_np_opt_count": 0,
    "percent_diff_st_mt": 0
}

BEST_MT_CACHE_JSON = {
    "workload": "",
    "cost": 0,
    "st_size": 0,
    "st_latency": 0,
    "mt_size": "",
    "mt_latency": "",
    "percent_diff": 0.0 
}

class PerCostAnalysis:
    def __init__(self, mt_label):
        self.cost_data_dir = COST_DATA_DIR
        self.mt_label = mt_label 
        self.data = {}
        self.min_length = math.inf
        self.max_length = 0 
        self.min_cost = 1

    def load_data(self, write_policy):
        data_dir = self.cost_data_dir.joinpath(self.mt_label, write_policy)
        for file_path in data_dir.iterdir():
            df = pd.read_csv(file_path, names=OPT_ROW_JSON.keys())
            workload_name = file_path.stem 
            self.data[workload_name] = df
            self.min_length = min(self.min_length, len(df))
            self.max_length = max(self.max_length, len(df))

    def st_mt_percent_diff(self, row):
        return 100*(row["st_latency"] - min(row["mt_p_latency"], row["mt_np_latency"]))/row["st_latency"]

    def data_for_each_cost(self, write_policy, mt_label):
        self.load_data(write_policy)
        cost_data = defaultdict(list)
        for cost_value in range(self.min_cost, self.min_cost+self.max_length):
            for workload_name in self.data:
                df = self.data[workload_name]
                cost_entry = df[df["cost"] == cost_value]
                if len(cost_entry) > 0:
                    row = cost_entry.iloc[0]
                    percent_diff = self.st_mt_percent_diff(row)
                    cost_data[cost_value].append(percent_diff)

        perf_stats_data = defaultdict(int)
        mean_array = []
        max_array = []
        errbar_array = []
        for cost in cost_data:
            stats_data = stats.describe(cost_data[cost])
            perf_stats_data[cost] = stats_data
            mean_array.append(stats_data[2])
            max_array.append(stats_data[1][1])
            errbar_array.append([stats_data[2]-stats_data[1][0], stats_data[1][1]-stats_data[2]])
            

        fig, ax = plt.subplots(figsize=(14,7))
        ax.plot(mean_array, "-*", label="Mean", markersize=10)
        ax.plot(max_array, "-8", label="Max", markersize=10)
        ax.set_ylabel("Percentage latency reduction from MT caching (%)")
        ax.set_xlabel("Cost ($)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("{}-{}.png".format(write_policy, mt_label))
        plt.close()

    def run(self):
        self.load_data("wb")
        for cost_value in range(self.min_cost, self.min_cost+self.max_length):
            st_mt_percent_diff_array = np.zeros(len(self.data.keys()), dtype=float)
            counter = Counter()
            best_mt_cache = BEST_MT_CACHE_JSON.copy()
            for index, key in enumerate(self.data):
                cur_df = self.data[key]
                entry = cur_df[cur_df["cost"] == cost_value]

                if len(entry) > 0:
                    row = entry.iloc[0]
                    st_mt_percent_diff_array[index] = self.st_mt_percent_diff(row)
                    counter[row["mt_opt_flag"]] += 1

                    if st_mt_percent_diff_array[index] > best_mt_cache["percent_diff"]:
                        best_mt_cache["percent_diff"] = st_mt_percent_diff_array[index]
                        best_mt_cache["st_size"] = row["st_size"]
                        best_mt_cache["st_latency"] = row["st_latency"]
                        best_mt_cache["mt_latency"] = min(row["mt_p_latency"], row["mt_np_latency"])
                        best_mt_cache["cost"] = row["cost"]
                        best_mt_cache["workload"] = key
                        best_mt_cache["mt_size"] = row["mt_p_size_array"] if row["mt_opt_flag"]==1 else row["mt_np_size_array"]

            print(best_mt_cache)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-workload analysis of cost analysis data for a workload.")
    parser.add_argument("--mt_label",
        default="D1_S1_H1",
        help="The label of the device type to be used.")
    args = parser.parse_args()

    analysis = PerCostAnalysis(args.mt_label)
    analysis.data_for_each_cost("wb", args.mt_label)
    analysis.data_for_each_cost("wt", args.mt_label)

    
