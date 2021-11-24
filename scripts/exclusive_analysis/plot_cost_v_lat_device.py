import argparse 
import pathlib 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from collections import defaultdict


def plot_cost_v_lat(cost_array_list, lat_reduced_array_list, label_array, output_path, log=0):

    plt.figure(figsize=[14, 10])
    plt.rcParams.update({'font.size': 25})
    ax = plt.subplot(1,1,1)

    line_style_list = ["--*", "--o", "--v", "--D", "--5"]

    line_style_dict = {
        "D1_D2_H1": "--*",
        "D1_D2_H2": "-.o",
        "D1_D2_S2": "..v",
        "D1_S1_H1": "--D",
        "D1_S1_H2": "--+",
        "D1_S1_S2": "--1",
        "D1_S2_H2": "-->",
        "D2_S1_H1": "--o",
        "D2_S1_H1": "--2",
        "D2_S1_H2": "--s",
        "D2_S2_H2": "--5",
    }

    for index, label_name in enumerate(label_array):
        line_style_dict[label_name] = line_style_list[index]

    print(cost_array_list)


    for index in range(len(cost_array_list)):
        cost_array = cost_array_list[index]
        print("THIS COST ARRAYT")
        print(cost_array)
        lat_reduced_array = lat_reduced_array_list[index]

        x = np.zeros(len(cost_array)+1)
        x[1:] = cost_array

        y = np.zeros(len(lat_reduced_array)+1)
        y[1:] = lat_reduced_array

        ax.plot(x, y, line_style_dict[label_array[index]], label=label_array[index], markersize=12, alpha=0.8)

    plt.xlabel("Scaled Purchase Cost ($)")
    plt.ylabel("Latency Reduced (ms)")

    if log:
        ax.set_yscale("log")
        plt.ylabel("log(Latency Reduced(ms))")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main(data_path, output_path, workload_name):

    label_data = defaultdict(list)
    cost_data = defaultdict(list)
    wt_lat_reduced_data = defaultdict(list)
    wb_lat_reduced_data = defaultdict(list)
    for device_cost_analysis_file in data_path.iterdir():
        device_config_name = device_cost_analysis_file.stem 
        if "st" not in device_config_name:
            df = pd.read_csv(device_cost_analysis_file, 
                names=["c", "wb_t1", "wb_t2", "wb_lat", "wb_d_lat", "wb_d_lat_per_dollar", "wt_t1", "wt_t2", "wt_lat", "wt_d_lat", "wt_d_lat_per_dollar"])

            label_data[device_config_name.split("_")[-1]].append(device_config_name)
            cost_data[device_config_name.split("_")[-1]].append(df["c"])
            wb_lat_reduced_data[device_config_name.split("_")[-1]].append(df["wb_d_lat"])
            wt_lat_reduced_data[device_config_name.split("_")[-1]].append(df["wt_d_lat"])

    print(cost_data.keys())

    for storage_backend in cost_data:
        cost_array_list = cost_data[storage_backend]
        wb_lat_reduced_array_list = wb_lat_reduced_data[storage_backend]
        wt_lat_reduced_array_list = wt_lat_reduced_data[storage_backend]

        output_file_name = "wb_{}_{}.png".format(storage_backend, workload_name)
        plot_cost_v_lat(cost_array_list, wb_lat_reduced_array_list, label_data[storage_backend], output_path.joinpath(output_file_name))

        output_file_name = "wb_log_{}_{}.png".format(storage_backend, workload_name)
        plot_cost_v_lat(cost_array_list, wb_lat_reduced_array_list, label_data[storage_backend], output_path.joinpath(output_file_name), log=1)

        output_file_name = "wt_{}_{}.png".format(storage_backend, workload_name)
        plot_cost_v_lat(cost_array_list, wt_lat_reduced_array_list, label_data[storage_backend], output_path.joinpath(output_file_name))

        output_file_name = "wt_log_{}_{}.png".format(storage_backend, workload_name)
        plot_cost_v_lat(cost_array_list, wt_lat_reduced_array_list, label_data[storage_backend], output_path.joinpath(output_file_name), log=1)

            
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Cost vs Latency for different devices for the same workload")
    parser.add_argument("workload_name", help="Name of the workload to plot")
    parser.add_argument("--o", type=pathlib.Path, default=pathlib.Path("./cost_v_lat_device"),
        help="Directory to output the plots")
    args = parser.parse_args()

    # create the output directory for this workload if it doesn't already exist 
    workload_cost_output_dir = args.o.joinpath(args.workload_name)
    workload_cost_output_dir.mkdir(parents=True, exist_ok=True)

    # generate path of data file 
    cost_data_dir = pathlib.Path("/research2/mtc/cp_traces/mascots/cost/{}/".format(args.workload_name))
    
    main(cost_data_dir, workload_cost_output_dir, args.workload_name)