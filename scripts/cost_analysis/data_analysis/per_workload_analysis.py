import argparse 
import pathlib 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from mascots.traceAnalysis.RDHistProfiler import OPT_ROW_JSON 


def plot_cost_vs_percent_diff(csv_path, output_path):
    """ Plot cost against percent difference between ST and MT cache, 
        also label the pyramidal and non-pyramidal MT configurations. 

    Parameters
    ----------
    csv_path : str 
        path to the file with cost analysis data 
    output_path : str 
        path to output the plot 
    """

    print("Plot cost vs percent diff from file : {}".format(csv_path))
    df = pd.read_csv(csv_path, names=OPT_ROW_JSON)
    cost_array = df["cost"] 
    percent_diff = 100*(df["st_latency"] - df[["mt_p_latency", "mt_np_latency"]].min(axis=1))/df["st_latency"]
    st_opt_index = percent_diff[percent_diff<=0].index
    mt_opt_index = percent_diff[percent_diff>0].index 

    mt_opt_df = df.iloc[mt_opt_index]
    mt_np_opt_index = mt_opt_df[mt_opt_df["mt_p_latency"] > mt_opt_df["mt_np_latency"]].index

    percent_diff_array = np.zeros(len(df), dtype=float)
    st_opt_cost, mt_p_opt_cost, mt_np_opt_cost = [], [], []
    for i, row in df.iterrows():
        st_latency = row["st_latency"]
        mt_p_latency = row["mt_p_latency"]
        mt_np_latency = row["mt_np_latency"]
        mt_opt_latency = min(mt_p_latency, mt_np_latency)
        percent_diff = 100*(st_latency-mt_opt_latency)/st_latency
        percent_diff_array[i] = percent_diff
        if st_latency <= mt_opt_latency:
            st_opt_cost.append([i, percent_diff])
        else:
            if mt_p_latency >= mt_np_latency:
                mt_np_opt_cost.append([i, percent_diff])
            else:
                mt_p_opt_cost.append([i, percent_diff])


    fig, ax = plt.subplots(figsize=(14,7))
    plt.rcParams.update({'font.size': 18})

    ax.plot(percent_diff_array)
    ax.scatter([_[0] for _ in st_opt_cost], [_[1] for _ in st_opt_cost], 
        s=120, alpha=0.75, marker="o", label="ST")
    ax.scatter([_[0] for _ in mt_p_opt_cost], [_[1] for _ in mt_p_opt_cost], 
        s=120, alpha=0.75, marker="D", label="MT-P")
    ax.scatter([_[0] for _ in mt_np_opt_cost], [_[1] for _ in mt_np_opt_cost], 
        s=120, alpha=0.75, marker="v", label="MT-NP")
    
    ax.set_xlabel("Cost ($)")
    ax.set_ylabel("Difference between ST and MT latency (%)")
    ax.set_ylim([-0.009, 1.15*max(percent_diff_array)])
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def print_opt_t1_t2_percent_diff(self):
    pass 


def main(data_dir, output_dir):
    for write_policy_dir in data_dir.iterdir():
        write_policy = write_policy_dir.name
        for data_file_path in write_policy_dir.iterdir():
            workload_name = data_file_path.stem
            output_file = "{}_{}.png".format(write_policy, workload_name)
            output_path = output_dir.joinpath(output_file)
            plot_cost_vs_percent_diff(data_file_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-workload analysis of cost analysis data for a workload.")
    parser.add_argument("mt_label", help="The label of MT configuration to evaluate")
    parser.add_argument("--cost_data_dir", 
        default=pathlib.Path("/research2/mtc/cp_traces/exhaustive_cost_analysis/"),
        type=pathlib.Path,
        help="Directory containing MT configuration files")
    parser.add_argument("--output_dir", 
        default=pathlib.Path("/research2/mtc/cp_traces/st_vs_mt_lat_plot"),
        type=pathlib.Path,
        help="Directory to output plots")
    args = parser.parse_args()

    args.output_dir.joinpath(args.mt_label).mkdir(exist_ok=True, parents=True)
    main(args.cost_data_dir.joinpath(args.mt_label), args.output_dir.joinpath(args.mt_label))