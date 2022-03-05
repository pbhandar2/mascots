import argparse, json, pathlib, math 
import numpy as np 
import matplotlib.pyplot as plt

from mascots.traceAnalysis.RDHistProfiler import RDHistProfiler, OPT_ROW_JSON 
from mascots.traceAnalysis.RDHistPlotter import RDHistPlotter


def plot_rw_hrc(workload_name, allocation_unit=256):
    experiment_config_file_path = "../../experiment_config.json"
    with open(experiment_config_file_path, "r") as f:
        experiment_config = json.load(f) 

    out_dir = pathlib.Path(experiment_config["rw_hrc_plot_dir"])
    out_dir.mkdir(exist_ok=True)

    # rd_hist_path = pathlib.Path(experiment_config["rd_hist_4k_dir"]).joinpath("{}.csv".format(workload_name))
    # profiler = RDHistProfiler(rd_hist_path)
    # plotter = RDHistPlotter(profiler)
    # out_path = out_dir.joinpath("{}.png".format(workload_name))
    # plotter.multi_hrc(out_path)
    # print("Plot done : {}".format(out_path))
    
    rd_hist_path = pathlib.Path(experiment_config["rd_hist_4k_dir"]).joinpath("{}.csv".format(workload_name))
    rd_hist_data = np.loadtxt(rd_hist_path, delimiter=",", dtype=int)
    cold_miss = np.copy(rd_hist_data[0]) 

    len_rd_hist = math.ceil(len(rd_hist_data)/allocation_unit)
    rd_hist = np.zeros((len_rd_hist+1, 2), dtype=int)
    rd_hist[1:] = np.add.reduceat(rd_hist_data[1:], 
                                    range(0, len(rd_hist_data), 
                                    allocation_unit))
    
    read_count = rd_hist[:, 0].sum() + cold_miss[0]
    write_count = rd_hist[:, 1].sum() + cold_miss[1]

    read_hrc = 1-rd_hist[:, 0].cumsum()/read_count 
    write_hrc = 1-rd_hist[:, 1].cumsum()/write_count 

    markeverycount = math.ceil(len(read_hrc)/95)

    print(len(read_hrc), len(write_hrc), markeverycount)

    fig, ax = plt.subplots(figsize=(14,7))

    x = np.zeros(len(read_hrc), dtype=float)
    for i in range(len(read_hrc)):
        x[i] = read_hrc[i]*4096/1e9

    ax.plot(read_hrc, "-o", 
            markersize=13, alpha=0.7, markevery=markeverycount, label="Read")
    ax.plot(write_hrc, "-*", 
            markersize=13, alpha=0.7, markevery=markeverycount, label="Write")

    xticks = range(0, len(read_hrc), 2000)
    xtick_labels = [str(x/1000) for x in xticks]

    print(xticks)
    print(xtick_labels)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel("Cache Size (GB)", fontsize=25)
    ax.set_ylabel("Hit Rate", fontsize=25)
    plt.legend(fontsize=25, markerscale=2.)
    plt.tight_layout()
    out_path = out_dir.joinpath("{}.pdf".format(workload_name))
    plt.savefig(out_path)
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot read/write HRC for a workload")
    parser.add_argument("workload_name", 
        help="The name of the workload to be evaluted")
    args = parser.parse_args()

    plot_rw_hrc(args.workload_name)