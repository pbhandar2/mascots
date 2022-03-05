import json, pathlib 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from collections import OrderedDict

from mascots.traceAnalysis.RDHistProfiler import OPT_ROW_JSON 

experiment_config_file_path = "../../../experiment_config.json"
with open(experiment_config_file_path, "r") as f:
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


def get_mt_p_np_percent_latency_reduction(write_policy):
    mt_config_list = config["priority_2_tier_mt_label_list"]
    mt_p_bars, mt_np_bars, mt_p_std_bars, mt_np_std_bars = [], [], [], []
    mt_p_max, mt_np_max = [], [] 
    for mt_label in mt_config_list:
        print("Computing {} {}".format(mt_label, write_policy))
        mt_p_lat_gain_array, mt_np_lat_gain_array = [], []
        data_dir = ex_out_dir = pathlib.Path(config["ex_cost_analysis_dir"]).joinpath(mt_label, write_policy)
        for workload_data_path in data_dir.iterdir():
            df = pd.read_csv(workload_data_path, names=OPT_ROW_JSON)
            for _, row in df.iterrows():
                st_latency = float(row["st_latency"])
                mt_p_latency = float(row["mt_p_latency"])
                mt_np_latency = float(row["mt_np_latency"])

                if st_latency > min(mt_p_latency, mt_np_latency):
                    if mt_p_latency < mt_np_latency:
                        percent_diff = 100*(st_latency-mt_p_latency)/st_latency
                        mt_p_lat_gain_array.append(percent_diff)
                    else:
                        percent_diff = 100*(st_latency - mt_np_latency)/st_latency
                        mt_np_lat_gain_array.append(percent_diff)

        mt_p_lat_gain_array = np.array(mt_p_lat_gain_array, dtype=float)
        mt_np_lat_gain_array = np.array(mt_np_lat_gain_array, dtype=float)
        mt_p_max.append(mt_p_lat_gain_array.max())
        mt_np_max.append(mt_np_lat_gain_array.max())
        mt_p_bars.append(mt_p_lat_gain_array.mean())
        mt_np_bars.append(mt_np_lat_gain_array.mean())
        mt_p_std_bars.append(mt_p_lat_gain_array.std())
        mt_np_std_bars.append(mt_np_lat_gain_array.std())

    mt_p_bars = np.array(mt_p_bars, dtype=float)
    mt_np_bars = np.array(mt_np_bars, dtype=float)
    mt_p_max = np.array(mt_p_max, dtype=float)
    mt_np_max = np.array(mt_np_max, dtype=float)

    argsort = mt_np_bars.argsort()
    mt_p_bars = mt_p_bars[argsort]
    mt_np_bars = mt_np_bars[argsort]
    mt_p_max = mt_p_max[argsort]
    mt_np_max = mt_np_max[argsort]

    # yerr_p = np.zeros((2, len(mt_p_std_bars)), dtype=float)
    # yerr_p[1, :] = np.array(mt_p_std_bars, dtype=float)
    # yerr_np = np.zeros((2, len(mt_np_std_bars)), dtype=float)
    # yerr_np[1, :] = np.array(mt_np_std_bars, dtype=float)

    # print(mt_np_bars)
    # print(min(mt_np_bars + yerr_np[1, :]))

    return np.array(mt_p_bars, dtype=float), np.array(mt_np_bars, dtype=float), [mt_p_std_bars, mt_np_std_bars], mt_p_max, mt_np_max

                
def get_cache_label_list():
    mt_config_list = config["priority_2_tier_mt_label_list"]
    label_replaced_mt_config_list = []
    for mt_config in mt_config_list:
        split_mt_config = mt_config.split("_")
        label_replaced_mt_config_list.append("\n".join([cache_name_replace_map[_] for _ in split_mt_config]))
    return label_replaced_mt_config_list


def plot_hist(ax, x, bar_set_1, bar_set_2, error_bars):
    width = 0.75
    ax.bar(x - width/2, bar_set_1, width, yerr=error_bars[0], label="MT-P", hatch='++', edgecolor="black",color="palegreen", capsize=5)
    ax.bar(x + width/2, bar_set_2, width,  yerr=error_bars[0], label="MT-NP", hatch='.0', edgecolor="black",color="slateblue", capsize=5)
    ax.margins(x=0.01)


def st_vs_mt_p_np():
    mt_label_list = get_cache_label_list()
    x = np.array([_*2 for _ in range(len(mt_label_list))])
    mt_p_bars, mt_np_bars, error_bars, mt_p_max, mt_np_max = get_mt_p_np_percent_latency_reduction("wb")
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(28,11))
    plot_hist(axs[0], x, mt_p_bars, mt_np_bars, error_bars)
    axs[0].scatter(x-0.375, mt_p_max, marker="*", s=200, label="Max MT-P")
    axs[0].scatter(x+0.375, mt_np_max, marker="o", s=200, label="Max MT-NP")
    axs[0].tick_params(axis='both', which='major', labelsize=21)
    axs[0].set_xticks([])
    axs[0].set_ylim(bottom=0)

    mt_p_bars, mt_np_bars, error_bars, mt_p_max, mt_np_max = get_mt_p_np_percent_latency_reduction("wt")
    plot_hist(axs[1], x, mt_p_bars, mt_np_bars, error_bars)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(mt_label_list)
    axs[1].tick_params(axis='y', which='major', labelsize=21)
    axs[1].tick_params(axis='x', which='major', labelsize=19)
    axs[1].scatter(x-0.375, mt_p_max, marker="*", s=200, label="Max MT-P")
    axs[1].scatter(x+0.375, mt_np_max, marker="o", s=200, label="Max MT-NP")
    axs[1].set_ylim(bottom=0)
    axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4, borderaxespad=0., fontsize=22)
    fig.text(0.0001, 0.5, "Mean Percent Latency Reduction (%)", va='center', rotation='vertical', fontsize=26)

    plt.tight_layout(pad=3)
    plt.savefig("st_vs_mt_p_np.pdf", bbox_inches="tight")
    plt.close()


def get_hmrc_latency_error():
    lat_error_header = ["T1", "T2", "TS", "mean-5", "std-5", "space-5", "mean-10", "std-10", "space-10", "mean-50", "std-50", "space-50"]
    wb_hmrc_lat_error_data = pd.read_csv("/research2/mtc/cp_traces/hmrc_error_per_step_size/err_wb_10_5_50.csv",
                                            names=lat_error_header)
    wt_hmrc_lat_error_data = pd.read_csv("/research2/mtc/cp_traces/hmrc_error_per_step_size/err_wt_10_5_50.csv",
                                            names=lat_error_header)
    print(wb_hmrc_lat_error_data)
    wb_data, wt_data = wb_hmrc_lat_error_data["mean-10"].to_numpy(), wt_hmrc_lat_error_data["mean-10"].to_numpy()
    wb_std, wt_std = wb_hmrc_lat_error_data["std-10"].to_numpy(), wt_hmrc_lat_error_data["std-10"].to_numpy()
    sorted_index = wb_data.argsort()
    return wb_data[sorted_index], wt_data[sorted_index], wb_std[sorted_index], wt_std[sorted_index]


def get_hmrc_classification_error():
    lat_error_header = ["MT", "err_5", "err_10", "err_50", "hit_potential", "miss_potential"]
    wb_hmrc_lat_error_data = pd.read_csv("/research2/mtc/cp_traces/hmrc_error_per_step_size/mt_miss_wb_10_5_50.csv",
                                            names=lat_error_header)
    wt_hmrc_lat_error_data = pd.read_csv("/research2/mtc/cp_traces/hmrc_error_per_step_size/mt_miss_wt_10_5_50.csv",
                                            names=lat_error_header)
    return wb_hmrc_lat_error_data["err_10"].to_numpy(), wt_hmrc_lat_error_data["err_10"].to_numpy()


def get_hmrc_potential():
    lat_error_header = ["MT", "err_5", "err_10", "err_50", "hit_potential", "miss_potential"]
    wb_hmrc_lat_error_data = pd.read_csv("/research2/mtc/cp_traces/hmrc_error_per_step_size/mt_miss_wb_10_5_50.csv",
                                            names=lat_error_header)
    wt_hmrc_lat_error_data = pd.read_csv("/research2/mtc/cp_traces/hmrc_error_per_step_size/mt_miss_wt_10_5_50.csv",
                                            names=lat_error_header)
    
    wb_hit_potential = wb_hmrc_lat_error_data["hit_potential"].to_numpy()
    wb_miss_potential = wb_hmrc_lat_error_data["miss_potential"].to_numpy()
    wt_hit_potential = wt_hmrc_lat_error_data["hit_potential"].to_numpy()
    wt_miss_potential = wt_hmrc_lat_error_data["miss_potential"].to_numpy()

    return wb_hit_potential, wb_miss_potential, wt_hit_potential, wt_miss_potential
    


def wb_wt_lat_error():
    mt_label_list = get_cache_label_list()
    x = np.array([_*2 for _ in range(len(mt_label_list))])
    wb_bars, wt_bars, wb_std, wt_std = get_hmrc_latency_error()

    # yerr_p = np.zeros((2, len(mt_p_std_bars)), dtype=float)
    # yerr_p[1, :] = np.array(mt_p_std_bars, dtype=float)
    # yerr_np = np.zeros((2, len(mt_np_std_bars)), dtype=float)
    # yerr_np[1, :] = np.array(mt_np_std_bars, dtype=float)

    fig, ax = plt.subplots(figsize=(28, 7))
    width = 0.75 
    ax.bar(x - width/2, wb_bars, width, yerr=wb_std, label="Write-Back", 
            hatch='++', edgecolor="black",color="palegreen", capsize=5)
    ax.bar(x + width/2, wt_bars, width, yerr=wt_std, label="Write-Through", 
            hatch='.0', edgecolor="black",color="slateblue", capsize=5)
    ax.margins(x=0.01)
    ax.set_ylim(bottom=0)

    ax.tick_params(axis='x', which='major', labelsize=19)
    ax.tick_params(axis='y', which='major', labelsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(mt_label_list)
    ax.set_ylabel("Mean Percent Latency Difference (%)", fontsize=22)

    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig("wb_wt_lat_error.pdf")
    plt.close()


def plot_classification_error():
    wb_err, wt_err = get_hmrc_classification_error()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.violinplot([wb_err, wt_err], showmedians=True, showextrema=True, showmeans=True, vert=False)
    ax.set_yticks([1,2])
    ax.set_yticklabels(["Write-Back", "Write-Through"], rotation=90, va="center")
    ax.tick_params(axis='x', which='major', labelsize=19)
    ax.tick_params(axis='y', which='major', labelsize=22)
    ax.set_xlabel("Classification Error (%)", fontsize=25)
    plt.tight_layout()
    plt.savefig("violin_class.png")
    plt.close()


def main():
    wb_hit, wb_miss, wt_hit, wt_miss = get_hmrc_potential()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.violinplot([wb_hit, wb_miss, wt_hit, wt_miss], showextrema=True, showmeans=True, vert=False)
    ax.set_yticks([1, 2,3,4])
    ax.set_yticklabels(["WB Hit", "WB Miss", "WT Hit", "WT Miss"], rotation=90, va="center")
    ax.tick_params(axis='x', which='major', labelsize=19)
    ax.tick_params(axis='y', which='major', labelsize=22)
    ax.set_xlabel("Latency Reduction Potential (%)", fontsize=22)
    plt.tight_layout()
    plt.savefig("violin_wb_wt_hit_miss.pdf")
    plt.close()

if __name__ == "__main__":
    wb_wt_lat_error()
    st_vs_mt_p_np()