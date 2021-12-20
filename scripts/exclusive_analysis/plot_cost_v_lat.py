import argparse 
import pathlib 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


def plot_cost_v_lat(cost_array, lat_reduced_array, output_path, log=0):
    plt.figure(figsize=[14, 10])
    plt.rcParams.update({'font.size': 30})
    ax = plt.subplot(1,1,1)

    x = np.zeros(len(cost_array)+1)
    x[1:] = cost_array

    y = np.zeros(len(lat_reduced_array)+1)
    y[1:] = lat_reduced_array

    ax.plot(x, y, "--*", markersize=15)

    plt.xlabel("Scaled Purchase Cost ($)")
    plt.ylabel("Latency Reduced (ms)")

    if log:
        ax.set_yscale("log")
        plt.ylabel("log(Latency Reduced(ms))")
    
    plt.tight_layout()
    plt.savefig(output_path)


def main(data_path, output_path):
    df = pd.read_csv(data_path, 
        names=["c", "wb_t1", "wb_t2", "wb_lat", "wb_d_lat", "wb_d_lat_per_dollar", "wt_t1", "wt_t2", "wt_lat", "wt_d_lat", "wt_d_lat_per_dollar"])

    workload_name = data_path.parents[0].name 
    output_file_name = "wb_{}.png".format(workload_name)
    plot_cost_v_lat(df["c"], df["wb_d_lat"], output_path.joinpath(output_file_name))

    output_file_name = "wb_log_{}.png".format(workload_name)
    plot_cost_v_lat(df["c"], df["wb_d_lat"], output_path.joinpath(output_file_name), log=1)

    output_file_name = "wt_{}.png".format(workload_name)
    plot_cost_v_lat(df["c"], df["wt_d_lat"], output_path.joinpath(output_file_name))

    output_file_name = "wt_log_{}.png".format(workload_name)
    plot_cost_v_lat(df["c"], df["wt_d_lat"], output_path.joinpath(output_file_name), log=1)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Cost vs Latency Reduced Curve for a workload")
    parser.add_argument("data_path", type=pathlib.Path, help="path to cost data of a workload")
    parser.add_argument("--o", type=pathlib.Path, default=pathlib.Path("./cost_v_lat"),
        help="Directory to output the plots")
    args = parser.parse_args()

    # create the output directory for this workload if it doesn't already exist 
    workload_cost_output_dir = args.o.joinpath(args.data_path.stem)
    workload_cost_output_dir.mkdir(parents=True, exist_ok=True)
    
    main(args.data_path, workload_cost_output_dir)