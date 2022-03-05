import argparse, json, pathlib 
from stat_max_hmrc import StatMaxHMRC
import matplotlib.pyplot as plt
import numpy as np 
from scipy import stats

import random 

with open("../../experiment_config.json", "r") as f:
    experiment_config = json.load(f)


def load_data(write_policy, step_size, num_points, alg="v2"):
    hmrc_analysis_dir = pathlib.Path(experiment_config["mhmrc_cost_analysis_dir"])
    ex_analysis_dir = pathlib.Path(experiment_config["ex_cost_analysis_dir"])
    data = {}
    f = open("error_{}_{}_{}.csv".format(step_size, num_points, write_policy), "w+")
    for mt_label_dir in hmrc_analysis_dir.iterdir():
        mt_label = mt_label_dir.name 
        
        if mt_label == "D1_D2_S2":
            continue 

        print("Evaluating {}".format(mt_label))

        error_info_dict = []
        error_array = []
        for workload_dir in mt_label_dir.joinpath(write_policy).iterdir():
            workload_name = workload_dir.name    

            exhaustive_dir = pathlib.Path(experiment_config["ex_cost_analysis_dir"])
            exhaustive_data_dir = exhaustive_dir.joinpath(mt_label, 
                                                        write_policy, 
                                                        "{}.csv".format(workload_name))
            
            if not exhaustive_data_dir.exists():
                continue 
            
            hmrc_analysis = StatMaxHMRC(mt_label, alg, step_size)
            hmrc_analysis.load_data(workload_name, write_policy)
            error_info_dict = hmrc_analysis.get_error(workload_name, write_policy, num_points_array=[num_points])
            error_array = []
            for cost_value in error_info_dict:
                cost_data = error_info_dict[cost_value]
                if num_points in cost_data:
                    error_entry = cost_data[num_points]
                    if error_entry[0] == 0 and error_entry[2]>0.0:
                        error_array.append(error_entry[2])
                    elif error_entry[0] == 1:
                        error_array.append(error_entry[2])
        
        data[mt_label] = error_array 
        mean_error = np.array(error_array).mean()
        print("MT: {}, write_policy: {}, Mean Percent Erorr: {}".format(mt_label, write_policy, mean_error))
        f.write("{},{},{},{}\n".format(mt_label, write_policy, num_points, mean_error))

    f.close()
    return data 


def plot_data(data, write_policy, num_points):
    mt_label_list = data.keys()
    storage_set = list(set([mt_label.split("_")[2] for mt_label in mt_label_list]))

    for storage_label in storage_set:
        fig, ax = plt.subplots(figsize=(14,7))
        x_axis = []
        for mt_label in data:
            if storage_label in mt_label:
                ax.violinplot(data[mt_label], 
                                showmeans=True)
                x_axis.append(mt_label)
        ax.set_xlabel("MT Cache", fontsize=22)
        ax.set_ylabel("Percent Error (%)", fontsize=22)
        plt.savefig("{}-{}-{}-voilin.png".format(write_policy, storage_label))
        plt.close()


def get_labels(mt_list):
    cache_name_replace_map = {
        "D1": "FastDRAM",
        "D3": "SlowDRAM",
        "H3": "SlowHDD",
        "S5": "SlowSSD",
        "S3": "MediumSSD",
        "S1": "FastSSD",
        "H1": "FastHDD"
    }
    label_list = []
    for mt_label in mt_list:
        split_mt_label = mt_label.split("_")
        new_label = [cache_name_replace_map[_] for _ in split_mt_label]
        label_list.append("\n".join(new_label))
    return label_list 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Violin plot of algorithm error.")
    parser.add_argument("--alg", 
        default="v2",
        help="The algorithm to analyze")
    parser.add_argument("--num_points", 
        default=1,
        type=int,
        help="Number of points evaluted")
    parser.add_argument("--step_size", 
        default=1,
        type=int,
        help="Step size for analysis")
    args = parser.parse_args()

    for write_policy in ["wb", "wt"]:
        data = load_data(write_policy, args.step_size, args.num_points)
        print("Loaded data for {}".format(write_policy))
        plot_data(data, write_policy, args.num_points)
        




