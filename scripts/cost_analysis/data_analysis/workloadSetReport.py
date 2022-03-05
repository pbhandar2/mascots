import pathlib, json, argparse, math
from re import X 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from collections import defaultdict 
from tabulate import tabulate

from mascots.traceAnalysis.MTConfigLib import MTConfigLib
from mascots.traceAnalysis.MHMRCProfiler import MHMRC_ROW_JSON
from mascots.traceAnalysis.RDHistProfiler import OPT_ROW_JSON 

class WorkloadSetReport:
    """ The class deals with analyzing a set of workload 
        and generating aggregate statistics and plots. 
    """

    def __init__(self,
            experiment_config="../../experiment_config.json"):
        
        self.mt_lib = MTConfigLib()
        with open(experiment_config, "r") as f:
            self.experiment_config = json.load(f) 
        
        self.ex_output_dir = pathlib.Path(self.experiment_config["ex_cost_analysis_dir"])
        # self.exhaustive_output_dir = pathlib.Path(self.experiment_config["opt_exhaustive_cost_analysis_dir"])
        self.two_tier_labels_batch_1 = self.experiment_config["priority_2_tier_mt_label_list"]
        self.device_config_dir = pathlib.Path(self.experiment_config["device_config_dir"]).joinpath("2")
        self.opt_cache_vs_mhmrc_dir = pathlib.Path(self.experiment_config["opt_cache_mhmrc_dir"])

        self.cache_name_replace_map = {
            "D1": "FastDRAM",
            "D3": "SlowDRAM",
            "H3": "SlowHDD",
            "S5": "SlowSSD",
            "S3": "MediumSSD",
            "S1": "FastSSD",
            "H1": "FastHDD"
        }

        self.total_np_opt_cache = defaultdict(int)
        self.total_cahce_evaluated = defaultdict(int)


    """ Find the highest difference in latency 
        based on device selection when 
        the storage is constant and where different 
        MT cache is best at different cost values. 
    """    
    def get_best_mt_cache(self, write_policy):

        mt_list = self.two_tier_labels_batch_1
        storage_device_list = list(set([_.split("_")[-1] for _ in mt_list]))
        workload_list = list([_.stem for _ in self.ex_output_dir.joinpath(mt_label, write_policy).iterdir()])
        per_workload_data = {}


        for storage_device in storage_device_list:
            for workload_name in workload_list:


                
                per_cost_data = {}
                for mt_label in mt_list:

                    split_mt_label = mt_label.split("_")
                    data_path = self.ex_output_dir.joinpath(mt_label, write_policy, "{}.csv".format(workload_name))
                    df = pd.read_csv(data_path, names=OPT_ROW_JSON)

                    for _, row in df.iterrows():
                        cost = row["cost"]

                        st_latency = row["st_latency"]
                        mt_p_latency = row["mt_p_latency"]
                        mt_np_latency = row["mt_np_latency"]

                        if st_latency < min(mt_np_latency, mt_p_latency):
                            opt_label = split_mt_label[0]
                        else:
                            if mt_p_latency < mt_np_latency:
                                opt_label = "-".join(split_mt_label[:2])










    def plot_opt_cache_type_vs_cost(self, mt_label, write_policy):
        """ Plot the percentage of different cache types at different 
            cost values. 

            Parameters
            ----------
            mt_label : str 
                the label to identify the MT cache 
            write_policy : str 
                "wb" (write back) or "wt" (write-through)
        """

        data_dir = self.ex_output_dir.joinpath(mt_label, write_policy)
        assert(data_dir.exists)

        # collect the count of MT types at various cost values 
        data_dict = defaultdict(lambda: defaultdict(int))
        for data_path in data_dir.iterdir():
            df = pd.read_csv(data_path, names=OPT_ROW_JSON)
            for _, row in df.iterrows():
                opt_mt_type = row[["st_latency", "mt_p_latency", "mt_np_latency"]].values.argmin()
                data_dict[row["cost"]][opt_mt_type] += 1

        # normalize the counts 
        hist_array = []
        for cost_limit in data_dict:
            cost_data = data_dict[cost_limit]
            mt_type_count = cost_data[0] + cost_data[1] + cost_data[2]
            if mt_type_count > 30:
                hist_array.append([cost_data[0]/mt_type_count, 
                                    cost_data[1]/mt_type_count, 
                                    cost_data[2]/mt_type_count])

        # plot the histogram 
        fig, ax = plt.subplots(figsize=(14,7))
        ax.bar(range(len(hist_array)), [_[0] for _ in hist_array], 1)
        ax.bar(range(len(hist_array)), [_[1] for _ in hist_array], 1, 
                bottom=[_[0] for _ in hist_array])
        bars = np.add([_[0] for _ in hist_array], [_[1] for _ in hist_array]).tolist()
        ax.bar(range(len(hist_array)), [_[2] for _ in hist_array], 1, 
                bottom=bars)
        plt.savefig("hist.png")
        plt.close()


    def plot_all_opt_cache_type_vs_scaled_cost(self):
        for mt_label in self.two_tier_labels_batch_1:
            for write_policy in ["wb", "wt"]:
                print("Plotting {}, {}".format(mt_label, write_policy))
                self.plot_opt_cache_type_vs_scaled_cost(mt_label, write_policy)


    def plot_opt_cache_type_vs_scaled_cost(self, 
            mt_label, 
            write_policy,
            bin_width=10):
        """ Plot the split of cost-optimal cache types at different 
            scaled cost value. 

            Parameters
            ----------
            mt_label : str 
                the label to identify the MT cache 
            write_policy : str 
                "wb" (write back) or "wt" (write-through)
            min_entry : int 
                minimum entries required per workload (optional) (Default: 10)
            min_height : int 
                the minimum number of 
            bin_width : int 
                the bin width representing scaled cost (1-100%) of the histogram (optional) (Default: 5)
        """

        x_axis, hist_array = self.get_opt_type_hist(mt_label, write_policy)

        # plot the histogram 
        fig, ax = plt.subplots(figsize=(14,7))
        ax.bar(x_axis, [_[0] for _ in hist_array], 
                                        bin_width, 
                                        align="edge",
                                        edgecolor="black", 
                                        hatch='x', 
                                        label="ST")
        ax.bar(x_axis, [_[1] for _ in hist_array],
                                        bin_width,
                                        align="edge",
                                        bottom=[_[0] for _ in hist_array],  
                                        edgecolor="black", 
                                        hatch='.0', 
                                        label="MT-P")
        bottom_bars = np.add([_[0] for _ in hist_array], [_[1] for _ in hist_array]).tolist()
        ax.bar(x_axis, [_[2] for _ in hist_array], 
                                        bin_width, 
                                        align="edge",
                                        bottom=bottom_bars,
                                        edgecolor="black", 
                                        hatch='++', 
                                        label="MT-NP")

        # labels and fontsize 
        ax.set_xlabel("Cost-Optimal Cache Type (%)", fontsize=22)
        ax.set_ylabel("Percentage (%)", fontsize=22)
        ax.set_xticks([_*bin_width for _ in range(1,11)])
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=15)

        output_dir = pathlib.Path(self.experiment_config["opt_cache_type_vs_scaled_cost_hist_dir"])
        output_file_name = "{}-{}-mt_type_vs_scaled_cost.png".format(mt_label, write_policy)
        output_path = output_dir.joinpath(output_file_name)
        print("Output plot path: {}".format(output_path))

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', fontsize=20, ncol=3, bbox_to_anchor=(0.5, 1.05))
        plt.tight_layout(pad=2)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


    def get_opt_type_hist(self, mt_label, write_policy, bin_width=10):
        data_dir = self.ex_output_dir.joinpath(mt_label, write_policy)
        # nested dict with key: cost range, opt_type and value is the count of the opt_type in that cost range
        data_dict = defaultdict(lambda: defaultdict(int))
        for data_path in data_dir.iterdir():
            df = pd.read_csv(data_path, names=OPT_ROW_JSON)

            for _, row in df.iterrows():
                st_latency = float(row["st_latency"])
                mt_p_latency = float(row["mt_p_latency"])
                mt_np_latency = float(row["mt_np_latency"])
                min_mt_latency = min(mt_p_latency, mt_np_latency)

                opt_mt_type = 0 
                if st_latency <= min_mt_latency:
                    opt_mt_type = 0
                else:
                    if mt_p_latency <= mt_np_latency:
                        opt_mt_type = 1 
                    else:
                        opt_mt_type = 2

                max_cost = int(row["max_cost"])
                cost_percent = 100*row["cost"]/max_cost
                if cost_percent > 100:
                    continue

                bucket = math.floor(cost_percent/bin_width)
                data_dict[bucket][opt_mt_type] += 1

        # normalize the counts 
        hist_array = []
        for cost_bucket in sorted(data_dict.keys()):
            cost_data = data_dict[cost_bucket]
            mt_type_count = cost_data[0] + cost_data[1] + cost_data[2]
            hist_array.append([100*cost_data[0]/mt_type_count, 
                                100*cost_data[1]/mt_type_count, 
                                100*cost_data[2]/mt_type_count])

            self.total_np_opt_cache[cost_bucket] += cost_data[2]
            self.total_cahce_evaluated[cost_bucket] += mt_type_count
        
        x_axis = [_*bin_width-(bin_width/2) for _ in range(1, len(hist_array)+1)]
        return x_axis, hist_array


    def multi_plot_opt_type_vs_scaled_cost(self, mt_label_grid, figure_label, out_file_path, bin_width=10):
        fig, axs = plt.subplots(nrows=len(mt_label_grid), ncols=len(mt_label_grid[0]), figsize=(14,6))
        #plt.subplots_adjust(wspace=0.1)
        for row_index, row in enumerate(mt_label_grid):
            for col_index, col in enumerate(row):
                x_axis, hist_array = self.get_opt_type_hist(col, "wb")
                axs[row_index, col_index].bar(x_axis, [_[0] for _ in hist_array], 
                                                bin_width, 
                                                align="center",
                                                edgecolor="black", 
                                                color="orange",
                                                hatch='x', 
                                                label="Single Tier")
                axs[row_index, col_index].bar(x_axis, [_[1] for _ in hist_array],
                                                bin_width,
                                                align="center",
                                                bottom=[_[0] for _ in hist_array],  
                                                edgecolor="black", 
                                                color="slateblue",
                                                hatch='.0', 
                                                label="Multi Tier-Pyramidal")
                bottom_bars = np.add([_[0] for _ in hist_array], [_[1] for _ in hist_array]).tolist()
                axs[row_index, col_index].bar(x_axis, [_[2] for _ in hist_array], 
                                                bin_width, 
                                                align="center",
                                                bottom=bottom_bars,
                                                edgecolor="black", 
                                                color="palegreen",
                                                hatch='++', 
                                                label="Multi Tier-NonPyramidal")
                axs[row_index, col_index].set_xticks([_*bin_width for _ in range(1,11)])
                axs[row_index, col_index].tick_params(axis='both', which='major', labelsize=10)

                mt_label_array = []
                for device_label in col.split("_"):
                    mt_label_array.append(self.cache_name_replace_map[device_label])
                mt_label = "-".join(mt_label_array)

                axs[row_index, col_index].set_title(mt_label, fontdict={'fontsize': 12})
                axs[row_index, col_index].set_xlabel("({})".format(figure_label[row_index][col_index]), fontsize=12)
                axs[row_index, col_index].set_xticks(x_axis)  

        axs[len(mt_label_grid)-1, len(mt_label_grid[0])-1].remove()
        fig.text(0.5, 0.02, "Percentage of Maximum Cost (%)", ha='center', fontsize=13)
        fig.text(0.0001, 0.5, "Classification of Cost-Optimal Tier Sizes (%)", va='center', rotation='vertical', fontsize=13)
        handles, labels = axs[-1][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', fontsize=12, bbox_to_anchor=(0.95, 0.1), title="Cost-Optimal Tier Sizes", title_fontsize=14)
        plt.tight_layout(pad=3)
        plt.savefig(out_file_path, bbox_inches="tight")
        plt.close()

        for cost_bucket in self.total_cahce_evaluated:
            np_count = self.total_np_opt_cache[cost_bucket]
            total = self.total_cahce_evaluated[cost_bucket]
            print("{}, {}/{}, {}".format(cost_bucket, np_count, total, 100*(np_count/total)))


    def init_opt_cache_type_output_dict(self, 
            workload_name, 
            mt_label, 
            og_ratio):
        return {
            "workload": workload_name,
            "mt_label": mt_label,
            "og_ratio": og_ratio,
            "tot": 0,
            "st": 0,
            "mt-p": 0,
            "mt-np": 0,
            "mean_np": 0.0,
            "mean_p": 0.0,
            "mean_mt": 0.0,
            "max_np": 0.0,
            "max_p": 0.0,
            "max_mt": 0.0
        }


    def generate_opt_cache_type_data(self, 
                write_policy,
                min_percent_latency_gain_array=[0,5,10,15,20]):
        """ For each MT cache, find the cost-optimal cache type and when the 
            cost-optimal cache type is MT, find the percent difference from 
            an ST cache. 

            Parameters
            ----------
            write_policy : str 
                "wb" write-back or "wt" write-through 
            min_percent_latency_gain_array : list 
                list of minimum percent latency difference for an MT cache to be optimal 
        """

        for min_percent_latency_gain in min_percent_latency_gain_array:
            for mt_label in self.two_tier_labels_batch_1:
                dict_array, mt_p_gain_array, mt_np_gain_array = [], [], []
                with open(self.device_config_dir.joinpath("{}.json".format(mt_label))) as f:
                    mt_config = json.load(f)
                og_ratio = self.mt_lib.get_overhead_gain_ratio(mt_config)
                ex_data_dir = self.ex_output_dir.joinpath(mt_label, write_policy)
                print("Evaluating MT label: {}".format(mt_label))
                for ex_data_path in ex_data_dir.iterdir():
                    workload_name = ex_data_path.stem
                    output_dict = self.init_opt_cache_type_output_dict(workload_name, 
                                                                        mt_label,
                                                                        og_ratio)
                    df = pd.read_csv(ex_data_path, names=OPT_ROW_JSON)

                    max_cost_value = int(df.iloc[0]["max_cost"])
                    max_cost_df = int(df["cost"].max())

                    if max_cost_df < max_cost_value:
                        print("Workload {} not ready!, {}/{}, {}".format(workload_name, max_cost_df, max_cost_value, max_cost_df/max_cost_value))
                        continue 
                    
                    # iterate each row of the DataFrame 
                    for row_index, row in df.iterrows():

                        cur_cost = int(row["cost"])
                        max_cost = int(row["max_cost"])

                        if cur_cost > max_cost:
                            continue 

                        st_latency = float(row["st_latency"])
                        mt_p_latency = float(row["mt_p_latency"])
                        mt_np_latency = float(row["mt_np_latency"])
                        if st_latency <= min(mt_p_latency, mt_np_latency):
                            output_dict["st"] += 1
                        elif st_latency > mt_p_latency and mt_p_latency < mt_np_latency:
                            percent_gain = 100*(st_latency - mt_p_latency)/st_latency
                            if percent_gain >= min_percent_latency_gain:
                                output_dict["mt-p"] += 1 
                                mt_p_gain_array.append(percent_gain)
                            else:
                                output_dict["st"] += 1
                        elif st_latency > mt_np_latency:
                            percent_gain = 100*(st_latency - mt_np_latency)/st_latency
                            if percent_gain >= min_percent_latency_gain:
                                output_dict["mt-np"] += 1 
                                mt_np_gain_array.append(percent_gain)
                            else:
                                output_dict["st"] += 1

                    if len(mt_p_gain_array) > 0:
                        output_dict["mean_p"] = np.array(mt_p_gain_array, dtype=float).mean()
                        output_dict["max_p"] = np.array(mt_p_gain_array, dtype=float).max()

                    if len(mt_np_gain_array) > 0:
                        output_dict["mean_np"] = np.array(mt_np_gain_array, dtype=float).mean()
                        output_dict["max_np"] = np.array(mt_np_gain_array, dtype=float).max()

                    mt_gain_array = mt_p_gain_array + mt_np_gain_array
                    if len(mt_gain_array) > 0:
                        output_dict["mean_mt"] = np.array(mt_gain_array, dtype=float).mean()
                        output_dict["max_mt"] = np.array(mt_gain_array, dtype=float).max()
                    output_dict["tot"] = output_dict["st"] + output_dict["mt-p"] + output_dict["mt-np"]
                    dict_array.append(output_dict)

                df = pd.DataFrame.from_dict(dict_array).sort_values(by=['workload'])
                output_file_name = "{}-{}.csv".format(mt_label, min_percent_latency_gain)
                output_file_path = self.opt_cache_vs_mhmrc_dir.joinpath(output_file_name)
                df.to_csv(output_file_path, index=False)
                print("Output path: {}".format(output_file_path))


    def normalize_data(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))


    def main_multi_plot_opt_type_vs_scaled_cost(self):
        mt_label_grid_1 = [["D1_D3_H3", "D1_D3_H1", "D1_D3_S5"], 
                           ["D1_S1_H3", "D1_S1_H1", "D1_S1_S5"],
                            ["D1_S3_H3", "D1_S3_H1"]]
        # mt_label_grid_1 = [["D1_D3_H3", "D1_D3_H1", "D1_D3_S5"], 
        #                     ["D1_S3_H3", "D1_S3_H1"]]
        output_file_name_1 = "D1-opt_type_vs_scaled_cost.pdf"
        figure_label = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]

        output_file_name_2 = "D2-opt_type_vs_scaled_cost.pdf"
        mt_label_grid_2 = [["D3_S1_H3", "D3_S1_H1", "D3_S1_S5"], ["D3_S3_H3", "D3_S3_H1"]]
        
        self.multi_plot_opt_type_vs_scaled_cost(mt_label_grid_1, figure_label, output_file_name_1)
        self.multi_plot_opt_type_vs_scaled_cost(mt_label_grid_2, [figure_label[0], figure_label[1]], output_file_name_2)

    
    def plot_mean_mhmrc_vs_mt_popularity(self, mt_label, threshold):
        """ A line plot of mean HMRC vs MT popularity. It could be for individual workloads 
            or across all workloads. 
        """
        mhmrc_df = pd.read_csv(pathlib.Path(self.experiment_config["general_data_dir"]).joinpath("4k_hmrc_stat.csv"),
                                    names=["workload", "total", "min", "max", "mean", "var", "skew", "kurt"])
        tot_count_dict = defaultdict(int)
        np_count_dict = defaultdict(int)
        mean_hmrc_dict = defaultdict(float)
        mhmrc_data_file = self.opt_cache_vs_mhmrc_dir.joinpath("{}-{}.csv".format(mt_label, threshold))
        file_threshold = int(mhmrc_data_file.stem.split("-")[1])
        if file_threshold == threshold:
            opt_type_df = pd.read_csv(mhmrc_data_file)
            for index, row in opt_type_df.iterrows():
                workload_name = row["workload"]
                if int(row["tot"] > 10):
                    hmrc_row = mhmrc_df[mhmrc_df["workload"]==workload_name]
                    np_count_dict[workload_name] += int(row["mt-np"]) + int(row["mt-p"])
                    tot_count_dict[workload_name] += int(row["tot"])
                    mean_hmrc_dict[workload_name] = float(hmrc_row["var"])
                else:
                    continue
        
        data_array = []
        for workload_name in np_count_dict:
            np_count = np_count_dict[workload_name]
            tot_count = tot_count_dict[workload_name]
            mean_hmrc = mean_hmrc_dict[workload_name]
            data_array.append([mean_hmrc, np_count/tot_count])
        
        data_array = sorted(data_array, key=lambda k: k[0])
        print(mhmrc_data_file.stem)
        print(data_array)

        fig, ax = plt.subplots(figsize=(14,7))
        ax.plot([_[0] for _ in data_array], [_[1] for _ in data_array], "-*")
        plt.savefig("hmrc_vs_mt.png")
                

    def plot_mean_mhmrc_vs_mean_mt_gain(self, mt_label, threshold):
        mhmrc_df = pd.read_csv(pathlib.Path(self.experiment_config["general_data_dir"]).joinpath("4k_hmrc_stat.csv"),
                                    names=["workload", "total", "min", "max", "mean", "var", "skew", "kurt"])
        mhmrc_data_file = self.opt_cache_vs_mhmrc_dir.joinpath("{}-{}.csv".format(mt_label, threshold))
        opt_type_df = pd.read_csv(mhmrc_data_file)
        data = []
        for index, row in opt_type_df.iterrows():
            workload_name = row["workload"]
            if int(row["tot"] > 10):
                hmrc_row = mhmrc_df[mhmrc_df["workload"]==workload_name]
                data.append([float(hmrc_row["max"]), row["max_mt"]])
            else:
                continue
        
        data = sorted(data, key=lambda k: k[0])

        fig, ax = plt.subplots(figsize=(14,7))
        ax.plot([_[0] for _ in data], [_[1] for _ in data], "-*")
        ax.set_xscale("log")
        plt.savefig("hmrc_vs_mean_mt.png")


    def sort_workloads_by_mt_popularity(self):
        mhmrc_df = pd.read_csv(pathlib.Path(self.experiment_config["general_data_dir"]).joinpath("4k_hmrc_stat.csv"),
                                    names=["workload", "total", "min", "max", "mean", "var", "skew", "kurt"])
        property_df = pd.read_csv(pathlib.Path(self.experiment_config["general_data_dir"]).joinpath("block_read_write_stats.csv"))
        threshold_array = [0,5,10,15,20]
        for threshold in [threshold_array[0]]:
            for mt_label in self.two_tier_labels_batch_1:
                if "D3" in mt_label:
                    continue 
                data_array = []
                mhmrc_data_file = self.opt_cache_vs_mhmrc_dir.joinpath("{}-{}.csv".format(mt_label, threshold))
                df = pd.read_csv(mhmrc_data_file)
                
                for index, row in df.iterrows():
                    workload_name = row["workload"]
                    mt_count = int(row["mt-p"]) + int(row["mt-np"])
                    tot = int(row["tot"])
                    mean_mt = float(row["mean_mt"])
                    max_mt = float(row["max_mt"])

                    if tot<5:
                        continue 

                    mhmrc_row = mhmrc_df[mhmrc_df["workload"]==workload_name]
                    property_row = property_df[property_df["workload"]==workload_name]

                    data_array.append([workload_name, mean_mt, property_row["read_ws"]/(property_row["read_ws"]+property_row["write_ws"])])
                    
                data_array = sorted(data_array, key=lambda k:k[1])
                print(mt_label)
                print(tabulate(data_array, headers=["w", "mean_mt", "r/w ratio"]))


    def get_max_latency_gain(self):
        pass 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a set of workloads")

    parser.add_argument("--mt_label", 
        default="D1_S1_H1",
        type=str,
        help="Label of the MT cache")

    parser.add_argument("--write_policy", 
        default="wb",
        type=str,
        help="Write policy to analyze")

    args = parser.parse_args()
    
    workloadSetAnalysis = WorkloadSetReport()

    # workloadSetAnalysis.plot_opt_cache_type_vs_cost(args.mt_label, args.write_policy)
    # workloadSetAnalysis.plot_all_opt_cache_type_vs_scaled_cost()
    # workloadSetAnalysis.plot_opt_cache_type_vs_scaled_cost("D3_S5_H3", "wb")
    # workloadSetAnalysis.generate_opt_cache_type_data("wb")
    workloadSetAnalysis.main_multi_plot_opt_type_vs_scaled_cost()

    # workloadSetAnalysis.sort_workloads_by_mt_popularity()

