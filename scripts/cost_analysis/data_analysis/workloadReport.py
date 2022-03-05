import json, pathlib, argparse, math
import pandas as pd 
import numpy as np 
from statistics import mean
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict

from mascots.traceAnalysis.MHMRCProfiler import MHMRC_ROW_JSON
from mascots.traceAnalysis.RDHistProfiler import OPT_ROW_JSON 

class WorkloadReport:
    def __init__(self, 
            workload_name,
            mt_label,
            write_policy,
            alg="v2",
            allocation_unit=256,
            experiment_config_file_path="../../experiment_config.json"):

        self.experiment_config_file_path = experiment_config_file_path
        with open(self.experiment_config_file_path, "r") as f:
            self.experiment_config = json.load(f) 

        self.cache_name_replace_map = {
            "D1": "FastDRAM",
            "D3": "SlowDRAM",
            "H3": "SlowHDD",
            "S5": "SlowSSD",
            "S3": "MediumSSD",
            "S1": "FastSSD",
            "H1": "FastHDD"
        }
        self.latency_label_array = ["st_latency", "mt_p_latency", "mt_np_latency"]
        self.rd_hist_path = pathlib.Path(self.experiment_config["rd_hist_4k_dir"]).joinpath("{}.csv".format(workload_name))



        self.workload_name = workload_name 
        self.mt_label = mt_label 
        self.write_policy = write_policy
        self.alg = alg 
        self.allocation_unit = allocation_unit 
        
        # output of the exhaustive search to find cost-efficient MT cache 
        self.ex_output_dir = pathlib.Path(self.experiment_config["ex_cost_analysis_dir"])
        self.load_data()
        self.marker_list = [">", "P", "8", "*", "x", "d", "|"]


        self.exhaustive_output_dir = pathlib.Path(self.experiment_config["opt_exhaustive_cost_analysis_dir"])
        self.exhaustive_file_path = self.ex_output_dir.joinpath(
                                        mt_label,
                                        write_policy,
                                        "{}.csv".format(self.workload_name))
        self.exhaustive_df = pd.read_csv(self.exhaustive_file_path, names=OPT_ROW_JSON)
        self.exhaustive_df.astype({
            "cost": int,
            "st_latency": float,
            "mt_p_latency": float,
            "mt_np_latency": float,
            "mt_opt_flag": int 
        })
        self.st_mt_output_dir = pathlib.Path(self.experiment_config["st_mt_latency_plots_dir"])
        self.hmrc_exhaustive_plot_dir = pathlib.Path(self.experiment_config["hmrc_exhaustive_plots_dir"])
        self.mhmrc_output_dir = pathlib.Path(self.experiment_config["mhmrc_cost_analysis_dir"])
        self.num_points_vs_accuracy_plot_dir = pathlib.Path(self.experiment_config["hmrc_eval_points_vs_accuracy_plots_dir"])
        self.points_filtered_plot_dir = pathlib.Path(self.experiment_config["points_filtered_plots_dir"])
        self.latency_per_dollar_plot_dir = pathlib.Path(self.experiment_config["latency_per_dollar_plots_dir"])

        self.hmrc_analysis_data = {}
        # for alg_step_size_dir in self.mhmrc_output_dir.joinpath(mt_label, 
        #                             write_policy, workload_name, alg).iterdir():
        #     for mhmrc_analysis_file in alg_step_size_dir.iterdir():
        #         if mhmrc_analysis_file.exists():
        #             self.hmrc_analysis_data[int(mhmrc_analysis_file.stem)] = pd.read_csv(mhmrc_analysis_file, 
        #                                                                 names=MHMRC_ROW_JSON)
        #             self.hmrc_analysis_data[int(mhmrc_analysis_file.stem)].astype({
        #                 "cost": int,
        #                 "latency": float
        #             })


    def get_opt_cache_type_flag(self, row):
        """ Get the OPT cache type given the output dict containing 
            latency of different cache types 

            Parameters
            ----------
            row : pd.Series
                Series representing a single row from output CSV 
            Return 
            ------
            opt_cache_type_flag : int 
                flag indicating the type of cache that was optimal 
        """

        st_latency = row["st_latency"]
        mt_p_latency = row["mt_p_latency"]
        mt_np_latency = row["mt_np_latency"]

        mt_opt_flag = 0
        if mt_p_latency < st_latency and mt_p_latency < mt_np_latency:
            mt_opt_flag = 1
        elif mt_np_latency < st_latency and mt_np_latency <= mt_p_latency:
            mt_opt_flag = 2
        return mt_opt_flag 


    def get_eval_points_vs_accuracy(self, cost_limit):
        """ Get the accuracy of the algorithm when different number of 
            points are evaluated. 

            Parameters
            ----------
            cost_limit : int
                the cost limit of the cache 
            
            Return 
            ------
            data : np.array 
                2D array with columns representing points evaluated and accuracy 
        """

        cache_type_array = ["st_latency", "mt_p_latency", "mt_np_latency"]
        data = []
        for key in self.hmrc_analysis_data:
            hmrc_data = self.hmrc_analysis_data[key]
            exhaustive_data = self.exhaustive_df
            if key > 0:
                exhaustive_row = exhaustive_data[exhaustive_data["cost"]==cost_limit]
                hmrc_row = hmrc_data[hmrc_data["cost"]==cost_limit]

                if len(exhaustive_row) == 1 and len(hmrc_row) == 1:
                    mt_opt_latency = exhaustive_row.iloc[0][cache_type_array].min(axis=0)
                    hmrc_opt_latency = hmrc_row.iloc[0]["latency"]
                    percent_error = 100*(hmrc_opt_latency-mt_opt_latency)/mt_opt_latency 
                    data.append([key, percent_error])
        return sorted(data, key=lambda k:k[0])


    def plot_st_mt_latency(self):
        """ Plot ST, MT-P and MT-NP latency for each cost 
            value 

            Parameters
            ----------
            output_path : str 
                path to the plot 
        """

        fig, ax = plt.subplots(figsize=(14,7))
        cost_array = self.exhaustive_df["cost"].values
        st_latency_array = self.exhaustive_df["st_latency"].values
        mt_p_latency_array = self.exhaustive_df["mt_p_latency"].values
        mt_np_latency_array = self.exhaustive_df["mt_np_latency"].values

        ax.plot(cost_array, st_latency_array, "-D", 
            markersize=10, alpha=0.7, color='r', label="ST")
        ax.plot(cost_array, mt_p_latency_array, "-P", 
            markersize=10, alpha=0.7, color='g', label="MT-P")
        ax.plot(cost_array, mt_np_latency_array, "-X", 
            markersize=10, alpha=0.7, color='b', label="MT-NP")
        ax.set_xlabel("Cost ($)", fontsize=15)
        ax.set_ylabel("Mean Latency (\u03BCs)", fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=15)
        plt.legend(fontsize=15)
        plt.tight_layout()
        workload_output_dir = self.st_mt_output_dir.joinpath(self.workload_name)
        workload_output_dir.mkdir(exist_ok=True)
        output_file_name = "{}_{}.png".format(self.mt_label,
                                            self.write_policy)
        plt.savefig(workload_output_dir.joinpath(output_file_name))
        plt.close()


    def all_num_points_vs_accuracy(self):
        """ Plot the accuracy of algorithm when evaluating various 
            number of points 
        """

        max_cost = self.hmrc_analysis_data[0]["cost"].max()
        print("Max cost: {}".format(max_cost))
        data_dict = defaultdict(list)
        for cost_limit in range(1, max_cost+1):
            print("Evaluating cost: {}".format(cost_limit))
            data = self.get_eval_points_vs_accuracy(cost_limit)
            self.plot_num_points_vs_accuracy(cost_limit)
            for eval_count, percent_error in data:
                data_dict[eval_count].append(percent_error)

        fig, ax = plt.subplots(figsize=(14,7))
        ax.plot(data_dict.keys(), 
                [mean(data_dict[k]) for k in data_dict],
                "-D")
        ax.set_xlabel("Points Evaluated", fontsize=15)
        ax.set_ylabel("Mean Percent Error (%)", fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=15)
        plt.legend(fontsize=15)
        plt.tight_layout()
        workload_output_dir = self.num_points_vs_accuracy_plot_dir.joinpath(self.workload_name)
        workload_output_dir.mkdir(exist_ok=True)
        output_file_name = "{}_{}.png".format(self.mt_label,
                                            self.write_policy)
        plt.savefig(workload_output_dir.joinpath(output_file_name))
        plt.close()


    def plot_num_points_vs_accuracy(self, cost_limit):
        """ Plot number of points vs accuracy for this workload 
        """

        fig, ax = plt.subplots(figsize=(14,7))
        data = self.get_eval_points_vs_accuracy(cost_limit)
        ax.plot([_[0] for _ in data], [_[1] for _ in data], "-D",
                label="${}".format(cost_limit))
        ax.set_xlabel("Points Evaluated", fontsize=15)
        ax.set_ylabel("Percent Error (%)", fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=15)
        plt.legend(fontsize=15)
        plt.tight_layout()
        workload_output_dir = self.num_points_vs_accuracy_plot_dir.joinpath(self.workload_name)
        workload_output_dir.mkdir(exist_ok=True)
        output_file_name = "{}_{}_{}.png".format(self.mt_label,
                                            self.write_policy,
                                            cost_limit)
        plt.savefig(workload_output_dir.joinpath(output_file_name))
        plt.close()


    def plot_points_filtered(self):
        """ Plot number of points filtered at various cost prices 
        """

        hmrc_data = self.hmrc_analysis_data[0]
        cost_array = hmrc_data["cost"].values
        percent_points_filtered = hmrc_data["percent_points_filtered"].values

        fig, ax = plt.subplots(figsize=(14,7))
        ax.plot(cost_array, percent_points_filtered, "-*")
        ax.set_xlabel("Cost ($)", fontsize=15)
        ax.set_ylabel("Percent Points Filtered (%)", fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=15)
        plt.legend(fontsize=15)
        plt.tight_layout()
        workload_output_dir = self.points_filtered_plot_dir.joinpath(self.workload_name)
        workload_output_dir.mkdir(exist_ok=True)
        output_file_name = "{}_{}_{}.png".format(self.workload_name, 
                                            self.mt_label,
                                            self.write_policy)
        plt.savefig(workload_output_dir.joinpath(output_file_name))
        plt.close()


    def plot_mean_bandwidth_per_dollar(self, allocation_unit=256):
        """ Plot mean bandwidth per dollar cost of the cache 
        """

        cost_array = self.exhaustive_df["cost"].values
        bandwidth_per_dollar_array = np.zeros(len(cost_array), dtype=float)
        for index, row in self.exhaustive_df.iterrows():
            mt_opt_cache_type = self.get_opt_cache_type_flag(row)
            if mt_opt_cache_type == 0:
                latency = row["st_latency"]
            elif mt_opt_cache_type == 1:
                latency = row["mt_p_latency"]
            elif mt_opt_cache_type == 2:
                latency = row["mt_np_latency"]
            else:
                raise ValueError("Unrecognized OPT cache type: {}".format(mt_opt_cache_type))

            bandwidth_per_dollar_array[index] = 1e6/(row["cost"]*latency*allocation_unit)

        fig, ax = plt.subplots(figsize=(14,7))
        ax.plot(cost_array, bandwidth_per_dollar_array, "-*")
        ax.set_xlabel("Cost ($)", fontsize=15)
        ax.set_ylabel("Mean Bandwidth (MB/s) per dollar ($))", fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=15)
        plt.legend(fontsize=15)
        plt.tight_layout()
        workload_output_dir = self.latency_per_dollar_plot_dir.joinpath(self.workload_name)
        workload_output_dir.mkdir(exist_ok=True)
        output_file_name = "{}_{}_{}.png".format(self.workload_name, 
                                            self.mt_label,
                                            self.write_policy)
        plt.savefig(workload_output_dir.joinpath(output_file_name))
        plt.close()


    def get_min_latency(self, row):
        mt_opt_flag = int(row["mt_opt_flag"])
        return row[self.latency_label_array][mt_opt_flag]

    
    def get_mt_label_list(self):
        return [_.stem for _ in self.ex_output_dir.iterdir()]


    def get_opt_cache_for_storage_and_cost(self, storage_label, cost_limit, write_policy):
        """ Get the label of the optimal cache for a given cost limit 

            Parameters
            ----------
            cost_limit : int 
                cost limit in dollars 
            write_policy : str 
                "wb" for write-back and "wt" for write-through 

            Return
            ------
            opt_cache_label : str 
                the label of the cost-optimal cache 
        """

        min_latency = math.inf 
        opt_cache_label = None 

        for mt_label in self.data[write_policy]:
            cur_storage_label = mt_label.split("_")[-1]
            if cur_storage_label != storage_label:
                continue 

            mt_label_data = self.data[write_policy][mt_label]
            rows = mt_label_data[mt_label_data["cost"] == cost_limit]

            if len(rows) > 0:
                row = rows.iloc[0]
            elif len(rows) == 0:
                row = mt_label_data[mt_label_data["cost"]==mt_label_data["cost"].max()].iloc[0]
            
            latency = self.get_min_latency(row)

            if latency < min_latency:
                min_latency = latency 
                device_label_array = mt_label.split("_")
                if int(row["mt_opt_flag"]) == 0:
                    opt_cache_label = device_label_array[0]
                else:
                    opt_cache_label = "_".join([device_label_array[0], device_label_array[1]])

        return opt_cache_label, min_latency


    def plot_best_mt_cache_per_cost(self, write_policy):
        output_dir = pathlib.Path(self.experiment_config["opt_cache_vs_cost_plot_dir"])
        for storage_label in self.storage_device_list:



            opt_mt_label_array = []
            min_latency_array = []
            for cost_limit in range(1, self.max_cost+1):
                opt_mt_label, min_latency = self.get_opt_cache_for_storage_and_cost(storage_label, 
                                                                                        cost_limit, 
                                                                                        write_policy)
                opt_mt_label_array.append(opt_mt_label)
                min_latency_array.append(min_latency)

            fig, ax = plt.subplots(figsize=(14,7))
            ax.plot(min_latency_array, "--")

            # scatter plot 
            scatter_dict = defaultdict(list)
            for index, mt_label in enumerate(opt_mt_label_array):
                scatter_dict[mt_label].append(index)

            for mt_label in scatter_dict:
                processed_mt_label = "-".join([self.cache_name_replace_map[_] for _ in mt_label.split("_")])
                for index in scatter_dict[mt_label]:
                    if index % 3 != 0:
                        continue 
                    ax.scatter(index, min_latency_array[index], 
                                marker=self.experiment_config["cache_color_marker_map"][mt_label]["marker"], 
                                c=self.experiment_config["cache_color_marker_map"][mt_label]["color"], 
                                label=processed_mt_label, 
                                s=300,
                                alpha=0.6)
            
            ax.set_xlabel("Cost ($)", fontsize=30, labelpad=10)
            ax.set_ylabel("Mean Latency (\u03BCs)", fontsize=30)

            ax.tick_params(axis='both', which='major', labelsize=25)

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), fontsize=20)

            output_file_name = "{}_{}_{}.png".format(self.workload_name, storage_label, write_policy)
            output_path = output_dir.joinpath(output_file_name)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

        
    def load_data(self):
        data = {}
        data["wb"], data["wt"] = {}, {}
        max_cost = 0 
        storage_device_set = set()
        for mt_label_output_dir in self.ex_output_dir.iterdir():

            mt_label = mt_label_output_dir.stem 

            if mt_label == "D1_D2_S2":
                continue 

            storage_device_label = mt_label.split("_")[-1]
            storage_device_set.add(storage_device_label)
            for write_policy in ["wb", "wt"]:
                # a subdir for each write policy, inside it is a CSV for each workload 
                data_path = mt_label_output_dir.joinpath(write_policy, 
                                                            "{}.csv".format(self.workload_name))
                df = pd.read_csv(data_path, names=OPT_ROW_JSON)
                data[write_policy][mt_label] = df 
                max_cost = max(max_cost, df["cost"].max())
        self.data = data 
        self.max_cost = max_cost 
        self.storage_device_list = list(storage_device_set)


    def plot_read_write_hrc(self):
        pass 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze per workload data")

    parser.add_argument("workload_name", 
        help="The name of the workload to be evaluted")

    parser.add_argument("--mt_label", 
        default="D1_S1_H1",
        type=str,
        help="Label of the MT cache")

    parser.add_argument("--write_policy", 
        default="wb",
        type=str,
        help="Write policy to analyze")

    args = parser.parse_args()
    workloadReporter = WorkloadReport(args.workload_name,
                                        args.mt_label,
                                        args.write_policy)
    
    # workloadReporter.plot_st_mt_latency()
    # workloadReporter.all_num_points_vs_accuracy()
    # workloadReporter.plot_points_filtered()
    # workloadReporter.plot_mean_bandwidth_per_dollar()

    workloadReporter.plot_best_mt_cache_per_cost("wb")
    workloadReporter.plot_best_mt_cache_per_cost("wt")
